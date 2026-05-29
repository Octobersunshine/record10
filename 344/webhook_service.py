import json
import hmac
import hashlib
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from queue import Queue, PriorityQueue
from dataclasses import dataclass, field

import requests
from flask import current_app
from sqlalchemy import and_, func

from models import db, WebhookSubscription, WebhookDeliveryLog, WebhookDeadLetter

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETRY_INTERVALS = [2, 5, 10]


@dataclass(order=True)
class RetryItem:
    next_attempt_time: float
    priority: int = field(compare=True)
    subscription_id: int = field(compare=False)
    event_type: str = field(compare=False)
    event_data: Dict[str, Any] = field(compare=False)
    callback_url: str = field(compare=False)
    secret_token: Optional[str] = field(compare=False, default=None)
    retry_count: int = field(compare=False, default=0)
    last_status_code: Optional[int] = field(compare=False, default=None)
    last_error_message: Optional[str] = field(compare=False, default=None)
    created_at: float = field(compare=False, default_factory=lambda: time.time())

    def get_retry_interval(self) -> int:
        idx = min(self.retry_count, len(RETRY_INTERVALS) - 1)
        return RETRY_INTERVALS[idx]

    def should_retry(self) -> bool:
        return self.retry_count < MAX_RETRIES


class WebhookService:
    def __init__(self, app=None):
        self.app = app
        self.executor = None
        self.retry_executor = None
        self.retry_queue: PriorityQueue[RetryItem] = PriorityQueue()
        self.retry_thread: Optional[threading.Thread] = None
        self.retry_stop_event = threading.Event()
        self._retry_lock = threading.Lock()
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        self.app = app
        max_workers = app.config.get('WEBHOOK_MAX_WORKERS', 10)
        retry_workers = app.config.get('WEBHOOK_RETRY_WORKERS', 5)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.retry_executor = ThreadPoolExecutor(max_workers=retry_workers)
        self._start_retry_processor()

    def _start_retry_processor(self):
        if self.retry_thread and self.retry_thread.is_alive():
            return
        self.retry_stop_event.clear()
        self.retry_thread = threading.Thread(
            target=self._retry_processor_loop,
            daemon=True,
            name='webhook-retry-processor'
        )
        self.retry_thread.start()
        logger.info("Webhook retry processor started")

    def _stop_retry_processor(self):
        self.retry_stop_event.set()
        if self.retry_thread:
            self.retry_thread.join(timeout=5)
            self.retry_thread = None
        logger.info("Webhook retry processor stopped")

    def _retry_processor_loop(self):
        while not self.retry_stop_event.is_set():
            try:
                self._process_retry_queue()
            except Exception as e:
                logger.error(f"Error in retry processor loop: {e}", exc_info=True)
            time.sleep(0.5)

    def _process_retry_queue(self):
        now = time.time()
        items_to_process: List[RetryItem] = []

        with self._retry_lock:
            while not self.retry_queue.empty():
                item = self.retry_queue.queue[0]
                if item.next_attempt_time <= now:
                    items_to_process.append(self.retry_queue.get())
                else:
                    break

        for item in items_to_process:
            self.retry_executor.submit(self._execute_retry, item)

    def _execute_retry(self, item: RetryItem):
        with self.app.app_context():
            subscription = WebhookSubscription.query.get(item.subscription_id)
            if not subscription:
                logger.warning(
                    f"Subscription {item.subscription_id} no longer exists, "
                    f"moving event to dead letter"
                )
                self._move_to_dead_letter(
                    item,
                    resolved=True,
                    resolution_note='Subscription no longer exists'
                )
                return

            if not subscription.is_active:
                logger.warning(
                    f"Subscription {item.subscription_id} is inactive, "
                    f"moving event to dead letter"
                )
                self._move_to_dead_letter(
                    item,
                    resolved=True,
                    resolution_note='Subscription is inactive'
                )
                return

            if subscription.callback_url != item.callback_url:
                item.callback_url = subscription.callback_url
                item.secret_token = subscription.secret_token

            if item.secret_token != subscription.secret_token:
                item.secret_token = subscription.secret_token

            log_entry = self._deliver_webhook_internal(
                subscription,
                item.event_type,
                item.event_data,
                retry_count=item.retry_count
            )

            if log_entry.status_code and 200 <= log_entry.status_code < 300:
                logger.info(
                    f"Retry succeeded for event {item.event_type} "
                    f"to {item.callback_url} after {item.retry_count + 1} attempts"
                )
            elif item.should_retry():
                self._enqueue_retry(item, log_entry)
            else:
                logger.error(
                    f"All retries exhausted for event {item.event_type} "
                    f"to {item.callback_url}, moving to dead letter"
                )
                self._move_to_dead_letter(item, log_entry)

    def _enqueue_retry(self, item: RetryItem, log_entry: WebhookDeliveryLog):
        item.retry_count += 1
        item.last_status_code = log_entry.status_code
        item.last_error_message = log_entry.error_message

        interval = item.get_retry_interval()
        item.next_attempt_time = time.time() + interval
        item.priority = item.retry_count

        with self._retry_lock:
            self.retry_queue.put(item)

        logger.info(
            f"Enqueued retry {item.retry_count}/{MAX_RETRIES} for "
            f"event {item.event_type} to {item.callback_url}, "
            f"next attempt in {interval}s"
        )

    def _move_to_dead_letter(
        self,
        item: RetryItem,
        log_entry: Optional[WebhookDeliveryLog] = None,
        resolved: bool = False,
        resolution_note: Optional[str] = None
    ):
        with self.app.app_context():
            dead_letter = WebhookDeadLetter(
                subscription_id=item.subscription_id,
                event_type=item.event_type,
                event_data=json.dumps(item.event_data, ensure_ascii=False),
                callback_url=item.callback_url,
                secret_token=item.secret_token,
                last_status_code=log_entry.status_code if log_entry else item.last_status_code,
                last_error_message=log_entry.error_message if log_entry else item.last_error_message,
                retry_count=item.retry_count,
                resolved=resolved,
                resolved_at=datetime.now(timezone.utc) if resolved else None,
                resolution_note=resolution_note
            )
            db.session.add(dead_letter)
            db.session.commit()
            logger.warning(
                f"Event {item.event_type} moved to dead letter (ID: {dead_letter.id}), "
                f"retry count: {item.retry_count}, resolved: {resolved}"
            )
            return dead_letter

    def create_subscription(
        self,
        event_type: str,
        callback_url: str,
        description: Optional[str] = None,
        secret_token: Optional[str] = None,
        is_active: bool = True
    ) -> WebhookSubscription:
        existing = WebhookSubscription.query.filter(
            and_(
                WebhookSubscription.event_type == event_type,
                WebhookSubscription.callback_url == callback_url
            )
        ).first()

        if existing:
            existing.description = description
            existing.secret_token = secret_token
            existing.is_active = is_active
            db.session.commit()
            return existing

        subscription = WebhookSubscription(
            event_type=event_type,
            callback_url=callback_url,
            description=description,
            secret_token=secret_token,
            is_active=is_active
        )
        db.session.add(subscription)
        db.session.commit()
        return subscription

    def get_subscription(self, subscription_id: int) -> Optional[WebhookSubscription]:
        return WebhookSubscription.query.get(subscription_id)

    def list_subscriptions(
        self,
        event_type: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[WebhookSubscription]:
        query = WebhookSubscription.query
        if event_type:
            query = query.filter(WebhookSubscription.event_type == event_type)
        if is_active is not None:
            query = query.filter(WebhookSubscription.is_active == is_active)
        return query.order_by(WebhookSubscription.created_at.desc()).all()

    def update_subscription(
        self,
        subscription_id: int,
        **kwargs
    ) -> Optional[WebhookSubscription]:
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            return None

        allowed_fields = {'event_type', 'callback_url', 'description', 'secret_token', 'is_active'}
        for key, value in kwargs.items():
            if key in allowed_fields and value is not None:
                setattr(subscription, key, value)

        db.session.commit()
        return subscription

    def delete_subscription(self, subscription_id: int) -> bool:
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            return False

        db.session.delete(subscription)
        db.session.commit()
        return True

    def get_active_subscribers(self, event_type: str) -> List[WebhookSubscription]:
        return WebhookSubscription.query.filter(
            and_(
                WebhookSubscription.event_type == event_type,
                WebhookSubscription.is_active == True
            )
        ).all()

    def generate_signature(self, secret_token: str, payload: str, timestamp: str) -> str:
        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret_token.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def _build_headers(
        self,
        event_type: str,
        payload: str,
        secret_token: Optional[str] = None
    ) -> Dict[str, str]:
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Event': event_type,
            'X-Webhook-Timestamp': timestamp,
            'X-Webhook-Delivery-Id': hashlib.md5(
                f"{event_type}{timestamp}{payload}".encode()
            ).hexdigest()[:16]
        }

        if secret_token:
            headers['X-Webhook-Signature'] = self.generate_signature(
                secret_token, payload, timestamp
            )

        return headers

    def _deliver_webhook_internal(
        self,
        subscription: WebhookSubscription,
        event_type: str,
        event_data: Dict[str, Any],
        retry_count: int = 0
    ) -> WebhookDeliveryLog:
        payload = json.dumps(event_data, ensure_ascii=False)
        headers = self._build_headers(event_type, payload, subscription.secret_token)

        log_entry = WebhookDeliveryLog(
            subscription_id=subscription.id,
            event_type=event_type,
            event_data=payload,
            callback_url=subscription.callback_url,
            retry_count=retry_count
        )

        start_time = time.time()

        try:
            response = requests.post(
                subscription.callback_url,
                data=payload.encode('utf-8'),
                headers=headers,
                timeout=DEFAULT_TIMEOUT
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            log_entry.response_time_ms = elapsed_ms
            log_entry.status_code = response.status_code
            log_entry.response_body = response.text[:1000]

            if 200 <= response.status_code < 300:
                logger.info(
                    f"Webhook delivered successfully to {subscription.callback_url} "
                    f"for event {event_type}, status: {response.status_code}, "
                    f"latency: {elapsed_ms}ms"
                )
            else:
                logger.error(
                    f"Webhook delivery failed to {subscription.callback_url} "
                    f"for event {event_type}, status: {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            log_entry.response_time_ms = elapsed_ms
            log_entry.error_message = str(e)[:500]
            logger.error(
                f"Webhook delivery exception to {subscription.callback_url} "
                f"for event {event_type}: {e}"
            )

        db.session.add(log_entry)
        db.session.commit()
        return log_entry

    def _deliver_webhook(
        self,
        subscription: WebhookSubscription,
        event_type: str,
        event_data: Dict[str, Any],
        retry_count: int = 0
    ) -> WebhookDeliveryLog:
        log_entry = self._deliver_webhook_internal(
            subscription, event_type, event_data, retry_count
        )

        if log_entry.status_code and 200 <= log_entry.status_code < 300:
            return log_entry

        needs_retry = (
            log_entry.status_code in RETRY_STATUS_CODES
            or log_entry.error_message is not None
        )

        if needs_retry and retry_count < MAX_RETRIES:
            retry_item = RetryItem(
                next_attempt_time=time.time() + RETRY_INTERVALS[retry_count],
                priority=retry_count + 1,
                subscription_id=subscription.id,
                event_type=event_type,
                event_data=event_data,
                callback_url=subscription.callback_url,
                secret_token=subscription.secret_token,
                retry_count=retry_count + 1,
                last_status_code=log_entry.status_code,
                last_error_message=log_entry.error_message
            )

            with self._retry_lock:
                self.retry_queue.put(retry_item)

            logger.info(
                f"Queued for retry (attempt {retry_count + 1}/{MAX_RETRIES}): "
                f"event {event_type} to {subscription.callback_url}, "
                f"next attempt in {RETRY_INTERVALS[retry_count]}s"
            )
        elif not (log_entry.status_code and 200 <= log_entry.status_code < 300):
            dead_letter = WebhookDeadLetter(
                subscription_id=subscription.id,
                event_type=event_type,
                event_data=json.dumps(event_data, ensure_ascii=False),
                callback_url=subscription.callback_url,
                secret_token=subscription.secret_token,
                last_status_code=log_entry.status_code,
                last_error_message=log_entry.error_message,
                retry_count=retry_count
            )
            db.session.add(dead_letter)
            db.session.commit()
            logger.warning(
                f"All retries exhausted, event {event_type} moved to dead letter "
                f"(ID: {dead_letter.id}) for {subscription.callback_url}"
            )

        return log_entry

    def trigger_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        wait_for_completion: bool = False
    ) -> List[WebhookDeliveryLog]:
        subscribers = self.get_active_subscribers(event_type)
        logger.info(
            f"Triggering event '{event_type}' with {len(subscribers)} subscribers"
        )

        if not subscribers:
            return []

        results: List[WebhookDeliveryLog] = []

        if wait_for_completion or len(subscribers) == 1:
            for subscriber in subscribers:
                log = self._deliver_webhook(subscriber, event_type, event_data)
                results.append(log)
        else:
            futures = []
            for subscriber in subscribers:
                future = self.executor.submit(
                    self._deliver_webhook, subscriber, event_type, event_data
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Webhook delivery task failed: {e}")

        return results

    def get_delivery_logs(
        self,
        subscription_id: Optional[int] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[WebhookDeliveryLog]:
        query = WebhookDeliveryLog.query
        if subscription_id:
            query = query.filter(WebhookDeliveryLog.subscription_id == subscription_id)
        if event_type:
            query = query.filter(WebhookDeliveryLog.event_type == event_type)
        return query.order_by(WebhookDeliveryLog.delivered_at.desc()).limit(limit).all()

    def get_event_types(self) -> List[str]:
        result = db.session.query(
            WebhookSubscription.event_type
        ).distinct().all()
        return [row[0] for row in result]

    def get_dead_letters(
        self,
        resolved: Optional[bool] = False,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[WebhookDeadLetter]:
        query = WebhookDeadLetter.query
        if resolved is not None:
            query = query.filter(WebhookDeadLetter.resolved == resolved)
        if event_type:
            query = query.filter(WebhookDeadLetter.event_type == event_type)
        return query.order_by(WebhookDeadLetter.failed_at.desc()).limit(limit).all()

    def get_dead_letter(self, dead_letter_id: int) -> Optional[WebhookDeadLetter]:
        return WebhookDeadLetter.query.get(dead_letter_id)

    def retry_dead_letter(self, dead_letter_id: int) -> Optional[WebhookDeadLetter]:
        dead_letter = self.get_dead_letter(dead_letter_id)
        if not dead_letter:
            return None

        subscription = WebhookSubscription.query.get(dead_letter.subscription_id)
        if not subscription:
            dead_letter.resolved = True
            dead_letter.resolved_at = datetime.now(timezone.utc)
            dead_letter.resolution_note = 'Subscription no longer exists'
            db.session.commit()
            return dead_letter

        if not subscription.is_active:
            dead_letter.resolved = True
            dead_letter.resolved_at = datetime.now(timezone.utc)
            dead_letter.resolution_note = 'Subscription is inactive'
            db.session.commit()
            return dead_letter

        try:
            event_data = json.loads(dead_letter.event_data)
        except json.JSONDecodeError:
            dead_letter.resolved = True
            dead_letter.resolved_at = datetime.now(timezone.utc)
            dead_letter.resolution_note = 'Invalid event data (cannot parse JSON)'
            db.session.commit()
            return dead_letter

        log_entry = self._deliver_webhook(
            subscription,
            dead_letter.event_type,
            event_data
        )

        if log_entry.status_code and 200 <= log_entry.status_code < 300:
            dead_letter.resolved = True
            dead_letter.resolved_at = datetime.now(timezone.utc)
            dead_letter.resolution_note = 'Manual retry succeeded'
            db.session.commit()
            logger.info(f"Dead letter {dead_letter_id} retried successfully")
        else:
            logger.warning(
                f"Manual retry of dead letter {dead_letter_id} failed, "
                f"queued for automatic retry"
            )

        return dead_letter

    def resolve_dead_letter(
        self,
        dead_letter_id: int,
        resolution_note: str
    ) -> Optional[WebhookDeadLetter]:
        dead_letter = self.get_dead_letter(dead_letter_id)
        if not dead_letter:
            return None

        dead_letter.resolved = True
        dead_letter.resolved_at = datetime.now(timezone.utc)
        dead_letter.resolution_note = resolution_note[:500]
        db.session.commit()
        logger.info(f"Dead letter {dead_letter_id} marked as resolved: {resolution_note}")
        return dead_letter

    def delete_dead_letter(self, dead_letter_id: int) -> bool:
        dead_letter = self.get_dead_letter(dead_letter_id)
        if not dead_letter:
            return False

        db.session.delete(dead_letter)
        db.session.commit()
        logger.info(f"Dead letter {dead_letter_id} deleted")
        return True

    def get_retry_queue_stats(self) -> Dict[str, Any]:
        with self._retry_lock:
            queue_size = self.retry_queue.qsize()
            items = list(self.retry_queue.queue)

        now = time.time()
        overdue = sum(1 for item in items if item.next_attempt_time <= now)
        retry_counts = {}
        for item in items:
            retry_counts[item.retry_count] = retry_counts.get(item.retry_count, 0) + 1

        return {
            'queue_size': queue_size,
            'overdue_items': overdue,
            'retry_counts': retry_counts,
            'max_retries': MAX_RETRIES,
            'retry_intervals': RETRY_INTERVALS
        }

    def verify_signature(
        self,
        secret_token: str,
        payload: str,
        timestamp: str,
        signature: str,
        tolerance_seconds: int = 300
    ) -> bool:
        try:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            msg_ts = int(timestamp)
            if abs(now_ts - msg_ts) > tolerance_seconds:
                logger.warning(
                    f"Signature verification failed: timestamp expired "
                    f"(msg_ts={msg_ts}, now_ts={now_ts}, diff={abs(now_ts - msg_ts)}s)"
                )
                return False

            expected = self.generate_signature(secret_token, payload, timestamp)
            if not hmac.compare_digest(expected, signature):
                logger.warning("Signature verification failed: signature mismatch")
                return False

            return True
        except (ValueError, TypeError) as e:
            logger.warning(f"Signature verification failed: {e}")
            return False

    def get_delivery_stats(
        self,
        subscription_id: Optional[int] = None,
        event_type: Optional[str] = None
    ) -> Dict[str, Any]:
        query = db.session.query(WebhookDeliveryLog)
        if subscription_id:
            query = query.filter(WebhookDeliveryLog.subscription_id == subscription_id)
        if event_type:
            query = query.filter(WebhookDeliveryLog.event_type == event_type)

        total = query.count()

        successful = query.filter(
            WebhookDeliveryLog.status_code >= 200,
            WebhookDeliveryLog.status_code < 300
        ).count()

        failed = query.filter(
            db.or_(
                db.and_(
                    WebhookDeliveryLog.status_code < 200,
                    WebhookDeliveryLog.status_code.isnot(None)
                ),
                WebhookDeliveryLog.status_code >= 300
            )
        ).count()

        errored = query.filter(
            WebhookDeliveryLog.error_message.isnot(None),
            WebhookDeliveryLog.error_message != ''
        ).count()

        latency_query = query.filter(
            WebhookDeliveryLog.response_time_ms.isnot(None)
        )
        latency_values = [r[0] for r in latency_query.with_entities(
            WebhookDeliveryLog.response_time_ms
        ).all()]

        if latency_values:
            sorted_values = sorted(latency_values)
            count = len(sorted_values)
            avg_latency = round(sum(sorted_values) / count, 2)
            min_latency = sorted_values[0]
            max_latency = sorted_values[-1]
            mid = count // 2
            if count % 2 == 0:
                p50_latency = round((sorted_values[mid - 1] + sorted_values[mid]) / 2, 2)
            else:
                p50_latency = sorted_values[mid]
        else:
            avg_latency = 0
            min_latency = 0
            max_latency = 0
            p50_latency = 0

        success_rate = round((successful / total) * 100, 2) if total > 0 else 0.0

        status_code_dist = dict(
            query.with_entities(
                WebhookDeliveryLog.status_code,
                func.count(WebhookDeliveryLog.id)
            ).filter(
                WebhookDeliveryLog.status_code.isnot(None)
            ).group_by(
                WebhookDeliveryLog.status_code
            ).all()
        )

        return {
            'total_deliveries': total,
            'successful': successful,
            'failed': failed,
            'errored': errored,
            'success_rate': success_rate,
            'latency': {
                'avg_ms': avg_latency,
                'min_ms': min_latency,
                'max_ms': max_latency,
                'p50_ms': p50_latency
            },
            'status_code_distribution': {
                str(k): v for k, v in status_code_dist.items()
            }
        }

    def replay_event(
        self,
        log_id: int
    ) -> Optional[WebhookDeliveryLog]:
        log_entry = WebhookDeliveryLog.query.get(log_id)
        if not log_entry:
            return None

        subscription = WebhookSubscription.query.get(log_entry.subscription_id)
        if not subscription:
            logger.error(f"Cannot replay log {log_id}: subscription {log_entry.subscription_id} not found")
            return None

        if not subscription.is_active:
            logger.error(f"Cannot replay log {log_id}: subscription {log_entry.subscription_id} is inactive")
            return None

        try:
            event_data = json.loads(log_entry.event_data)
        except json.JSONDecodeError:
            logger.error(f"Cannot replay log {log_id}: invalid event data")
            return None

        new_log = self._deliver_webhook(
            subscription,
            log_entry.event_type,
            event_data
        )

        logger.info(
            f"Replayed event from log {log_id}: event_type={log_entry.event_type}, "
            f"new_log_id={new_log.id}"
        )
        return new_log

    def replay_events_by_type(
        self,
        event_type: str,
        limit: int = 100
    ) -> List[WebhookDeliveryLog]:
        logs = WebhookDeliveryLog.query.filter(
            WebhookDeliveryLog.event_type == event_type
        ).order_by(
            WebhookDeliveryLog.delivered_at.desc()
        ).limit(limit).all()

        results: List[WebhookDeliveryLog] = []
        for log_entry in logs:
            subscription = WebhookSubscription.query.get(log_entry.subscription_id)
            if not subscription or not subscription.is_active:
                continue

            try:
                event_data = json.loads(log_entry.event_data)
            except json.JSONDecodeError:
                continue

            new_log = self._deliver_webhook(subscription, event_type, event_data)
            results.append(new_log)

        logger.info(
            f"Replayed {len(results)} events for type {event_type}"
        )
        return results
