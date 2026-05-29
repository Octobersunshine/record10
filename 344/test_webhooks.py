import json
import hmac
import hashlib
import time
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from app import create_app, webhook_service
from models import db, WebhookSubscription, WebhookDeliveryLog, WebhookDeadLetter
from webhook_service import RetryItem, RETRY_INTERVALS, MAX_RETRIES


class WebhookAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_health_check(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'ok')

    def test_create_webhook_success(self):
        payload = {
            'event_type': 'user.created',
            'callback_url': 'https://example.com/webhooks/user-created',
            'description': 'Notify when a new user is created',
            'secret_token': 'my-secret-token-123'
        }
        response = self.client.post(
            '/api/webhooks',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertIn('data', data)
        self.assertEqual(data['data']['event_type'], 'user.created')
        self.assertEqual(
            data['data']['callback_url'],
            'https://example.com/webhooks/user-created'
        )
        self.assertTrue(data['data']['is_active'])
        self.assertIsNotNone(data['data']['id'])

    def test_create_webhook_missing_fields(self):
        payload = {'event_type': 'user.created'}
        response = self.client.post(
            '/api/webhooks',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_create_webhook_invalid_url(self):
        payload = {
            'event_type': 'user.created',
            'callback_url': 'not-a-valid-url'
        }
        response = self.client.post(
            '/api/webhooks',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_create_webhook_duplicate(self):
        payload = {
            'event_type': 'user.created',
            'callback_url': 'https://example.com/webhook'
        }
        self.client.post(
            '/api/webhooks',
            data=json.dumps(payload),
            content_type='application/json'
        )
        response = self.client.post(
            '/api/webhooks',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)

    def test_list_webhooks(self):
        for i in range(3):
            self.client.post(
                '/api/webhooks',
                data=json.dumps({
                    'event_type': f'event.{i}',
                    'callback_url': f'https://example.com/webhook{i}'
                }),
                content_type='application/json'
            )

        response = self.client.get('/api/webhooks')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['count'], 3)
        self.assertEqual(len(data['data']), 3)

    def test_list_webhooks_filter_by_event_type(self):
        self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook1'
            }),
            content_type='application/json'
        )
        self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'order.paid',
                'callback_url': 'https://example.com/webhook2'
            }),
            content_type='application/json'
        )

        response = self.client.get('/api/webhooks?event_type=user.created')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['count'], 1)
        self.assertEqual(data['data'][0]['event_type'], 'user.created')

    def test_get_webhook_by_id(self):
        create_response = self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook'
            }),
            content_type='application/json'
        )
        webhook_id = create_response.get_json()['data']['id']

        response = self.client.get(f'/api/webhooks/{webhook_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['data']['id'], webhook_id)

    def test_get_webhook_not_found(self):
        response = self.client.get('/api/webhooks/999')
        self.assertEqual(response.status_code, 404)

    def test_update_webhook(self):
        create_response = self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook',
                'is_active': True
            }),
            content_type='application/json'
        )
        webhook_id = create_response.get_json()['data']['id']

        response = self.client.put(
            f'/api/webhooks/{webhook_id}',
            data=json.dumps({
                'description': 'Updated description',
                'is_active': False
            }),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['data']['description'], 'Updated description')
        self.assertFalse(data['data']['is_active'])

    def test_delete_webhook(self):
        create_response = self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook'
            }),
            content_type='application/json'
        )
        webhook_id = create_response.get_json()['data']['id']

        response = self.client.delete(f'/api/webhooks/{webhook_id}')
        self.assertEqual(response.status_code, 200)

        get_response = self.client.get(f'/api/webhooks/{webhook_id}')
        self.assertEqual(get_response.status_code, 404)

    def test_list_event_types(self):
        self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook1'
            }),
            content_type='application/json'
        )
        self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'order.paid',
                'callback_url': 'https://example.com/webhook2'
            }),
            content_type='application/json'
        )

        response = self.client.get('/api/events')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['count'], 2)
        self.assertIn('user.created', data['data'])
        self.assertIn('order.paid', data['data'])

    @patch('webhook_service.requests.post')
    def test_trigger_event(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook',
                'secret_token': 'test-secret'
            }),
            content_type='application/json'
        )

        event_data = {
            'user_id': 123,
            'email': 'user@example.com',
            'username': 'testuser'
        }
        response = self.client.post(
            '/api/events/user.created',
            data=json.dumps({'data': event_data, 'wait_for_completion': True}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['subscriber_count'], 1)
        self.assertEqual(data['successful_deliveries'], 1)
        self.assertEqual(data['failed_deliveries'], 0)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], 'https://example.com/webhook')

        headers = call_args[1]['headers']
        self.assertIn('X-Webhook-Event', headers)
        self.assertIn('X-Webhook-Signature', headers)
        self.assertIn('X-Webhook-Timestamp', headers)

    def test_trigger_event_no_subscribers(self):
        response = self.client.post(
            '/api/events/nonexistent.event',
            data=json.dumps({'data': {'key': 'value'}}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['subscriber_count'], 0)

    @patch('webhook_service.requests.post')
    def test_get_delivery_logs(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        create_response = self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook'
            }),
            content_type='application/json'
        )
        webhook_id = create_response.get_json()['data']['id']

        self.client.post(
            '/api/events/user.created',
            data=json.dumps({'data': {'user_id': 1}, 'wait_for_completion': True}),
            content_type='application/json'
        )

        response = self.client.get(f'/api/webhooks/{webhook_id}/logs')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreaterEqual(data['count'], 1)

    def test_signature_generation(self):
        with self.app.app_context():
            secret = 'test-secret-key'
            payload = '{"test": "data"}'
            timestamp = str(int(datetime.now(timezone.utc).timestamp()))

            signature = webhook_service.generate_signature(secret, payload, timestamp)
            self.assertTrue(signature.startswith('sha256='))

            expected = hmac.new(
                secret.encode('utf-8'),
                f"{timestamp}.{payload}".encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            self.assertEqual(signature, f"sha256={expected}")

    @patch('webhook_service.requests.post')
    def test_webhook_retry_on_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Server Error'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            while not webhook_service.retry_queue.empty():
                webhook_service.retry_queue.get()

            result = webhook_service._deliver_webhook(
                subscription,
                'test.event',
                {'key': 'value'}
            )

            self.assertEqual(result.status_code, 500)
            self.assertEqual(result.retry_count, 0)
            self.assertEqual(mock_post.call_count, 1)
            self.assertEqual(webhook_service.retry_queue.qsize(), 1)

            retry_item = webhook_service.retry_queue.get()
            self.assertEqual(retry_item.retry_count, 1)
            self.assertEqual(retry_item.last_status_code, 500)

    def test_create_webhook_inactive(self):
        payload = {
            'event_type': 'user.created',
            'callback_url': 'https://example.com/webhook',
            'is_active': False
        }
        response = self.client.post(
            '/api/webhooks',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertFalse(data['data']['is_active'])

    def test_get_all_logs(self):
        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook'
            )
            db.session.add(subscription)
            db.session.commit()

            log = WebhookDeliveryLog(
                subscription_id=subscription.id,
                event_type='test.event',
                event_data='{}',
                callback_url='https://example.com/webhook',
                status_code=200
            )
            db.session.add(log)
            db.session.commit()

        response = self.client.get('/api/logs')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['count'], 1)

    def test_retry_intervals_configuration(self):
        self.assertEqual(len(RETRY_INTERVALS), 3)
        self.assertEqual(RETRY_INTERVALS[0], 2)
        self.assertEqual(RETRY_INTERVALS[1], 5)
        self.assertEqual(RETRY_INTERVALS[2], 10)
        self.assertEqual(MAX_RETRIES, 3)

    def test_retry_item_creation(self):
        item = RetryItem(
            next_attempt_time=time.time() + 10,
            priority=1,
            subscription_id=1,
            event_type='test.event',
            event_data={'key': 'value'},
            callback_url='https://example.com/webhook',
            secret_token='secret',
            retry_count=0
        )
        self.assertEqual(item.get_retry_interval(), 2)
        self.assertTrue(item.should_retry())
        item.retry_count = 1
        self.assertEqual(item.get_retry_interval(), 5)
        item.retry_count = 2
        self.assertEqual(item.get_retry_interval(), 10)
        item.retry_count = 3
        self.assertFalse(item.should_retry())

    def test_retry_item_priority_ordering(self):
        now = time.time()
        item1 = RetryItem(
            next_attempt_time=now + 10,
            priority=1,
            subscription_id=1,
            event_type='test.event',
            event_data={},
            callback_url='https://example.com/1',
            secret_token=None
        )
        item2 = RetryItem(
            next_attempt_time=now + 5,
            priority=2,
            subscription_id=2,
            event_type='test.event',
            event_data={},
            callback_url='https://example.com/2',
            secret_token=None
        )
        item3 = RetryItem(
            next_attempt_time=now + 5,
            priority=1,
            subscription_id=3,
            event_type='test.event',
            event_data={},
            callback_url='https://example.com/3',
            secret_token=None
        )

        self.assertTrue(item2 < item1)
        self.assertTrue(item3 < item2)
        self.assertTrue(item3 < item1)

    @patch('webhook_service.requests.post')
    def test_failed_delivery_queued_for_retry(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            while not webhook_service.retry_queue.empty():
                webhook_service.retry_queue.get()

            initial_queue_size = webhook_service.retry_queue.qsize()

            result = webhook_service._deliver_webhook(
                subscription,
                'test.event',
                {'user_id': 123}
            )

            self.assertEqual(result.status_code, 500)
            self.assertEqual(webhook_service.retry_queue.qsize(), initial_queue_size + 1)

            retry_item = webhook_service.retry_queue.get()

            self.assertIsNotNone(retry_item)
            self.assertEqual(retry_item.event_type, 'test.event')
            self.assertEqual(retry_item.retry_count, 1)
            self.assertEqual(retry_item.last_status_code, 500)

    @patch('webhook_service.requests.post')
    def test_all_retries_exhausted_moves_to_dead_letter(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = 'Persistent Error'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            dead_letter_count_before = WebhookDeadLetter.query.filter_by(
                subscription_id=subscription.id
            ).count()

            retry_item = RetryItem(
                next_attempt_time=time.time(),
                priority=3,
                subscription_id=subscription.id,
                event_type='test.event',
                event_data={'user_id': 123},
                callback_url='https://example.com/webhook',
                secret_token=None,
                retry_count=3,
                last_status_code=500,
                last_error_message='Persistent Error'
            )

            webhook_service._execute_retry(retry_item)

            dead_letter_count_after = WebhookDeadLetter.query.filter_by(
                subscription_id=subscription.id
            ).count()

            self.assertEqual(dead_letter_count_after, dead_letter_count_before + 1)

            dead_letter = WebhookDeadLetter.query.filter_by(
                subscription_id=subscription.id
            ).first()

            self.assertIsNotNone(dead_letter)
            self.assertEqual(dead_letter.event_type, 'test.event')
            self.assertEqual(dead_letter.last_status_code, 500)
            self.assertEqual(dead_letter.retry_count, 3)
            self.assertFalse(dead_letter.resolved)

    @patch('webhook_service.requests.post')
    def test_retry_succeeds(self, mock_post):
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.text = 'OK'

        mock_post.return_value = mock_response_success

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            retry_item = RetryItem(
                next_attempt_time=time.time(),
                priority=1,
                subscription_id=subscription.id,
                event_type='test.event',
                event_data={'user_id': 123},
                callback_url='https://example.com/webhook',
                secret_token=None,
                retry_count=1,
                last_status_code=500
            )

            webhook_service._execute_retry(retry_item)

            self.assertEqual(mock_post.call_count, 1)

            log = WebhookDeliveryLog.query.filter_by(
                subscription_id=subscription.id
            ).order_by(WebhookDeliveryLog.delivered_at.desc()).first()

            self.assertIsNotNone(log)
            self.assertEqual(log.status_code, 200)

            dead_letter_count = WebhookDeadLetter.query.filter_by(
                subscription_id=subscription.id
            ).count()
            self.assertEqual(dead_letter_count, 0)

    @patch('webhook_service.requests.post')
    def test_dead_letter_api_endpoints(self, mock_post):
        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            dead_letter = WebhookDeadLetter(
                subscription_id=subscription.id,
                event_type='test.event',
                event_data='{"user_id": 123}',
                callback_url='https://example.com/webhook',
                last_status_code=500,
                last_error_message='Internal Server Error',
                retry_count=3
            )
            db.session.add(dead_letter)
            db.session.commit()
            dead_letter_id = dead_letter.id

        list_response = self.client.get('/api/dead-letters')
        self.assertEqual(list_response.status_code, 200)
        list_data = list_response.get_json()
        self.assertGreaterEqual(list_data['count'], 1)

        get_response = self.client.get(f'/api/dead-letters/{dead_letter_id}')
        self.assertEqual(get_response.status_code, 200)
        get_data = get_response.get_json()
        self.assertEqual(get_data['data']['id'], dead_letter_id)
        self.assertEqual(get_data['data']['event_type'], 'test.event')

        get_not_found = self.client.get('/api/dead-letters/99999')
        self.assertEqual(get_not_found.status_code, 404)

        resolve_response = self.client.post(
            f'/api/dead-letters/{dead_letter_id}/resolve',
            data=json.dumps({'resolution_note': 'Manually processed'}),
            content_type='application/json'
        )
        self.assertEqual(resolve_response.status_code, 200)
        resolve_data = resolve_response.get_json()
        self.assertTrue(resolve_data['data']['resolved'])
        self.assertEqual(resolve_data['data']['resolution_note'], 'Manually processed')

        list_resolved = self.client.get('/api/dead-letters?resolved=true')
        self.assertEqual(list_resolved.status_code, 200)
        self.assertGreaterEqual(list_resolved.get_json()['count'], 1)

        list_unresolved = self.client.get('/api/dead-letters?resolved=false')
        self.assertEqual(list_unresolved.status_code, 200)
        self.assertEqual(list_unresolved.get_json()['count'], 0)

        delete_response = self.client.delete(f'/api/dead-letters/{dead_letter_id}')
        self.assertEqual(delete_response.status_code, 200)

        get_after_delete = self.client.get(f'/api/dead-letters/{dead_letter_id}')
        self.assertEqual(get_after_delete.status_code, 404)

    @patch('webhook_service.requests.post')
    def test_dead_letter_retry_api(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            dead_letter = WebhookDeadLetter(
                subscription_id=subscription.id,
                event_type='test.event',
                event_data='{"user_id": 123}',
                callback_url='https://example.com/webhook',
                last_status_code=500,
                last_error_message='Internal Server Error',
                retry_count=3
            )
            db.session.add(dead_letter)
            db.session.commit()
            dead_letter_id = dead_letter.id

        retry_response = self.client.post(
            f'/api/dead-letters/{dead_letter_id}/retry'
        )
        self.assertEqual(retry_response.status_code, 200)
        retry_data = retry_response.get_json()
        self.assertTrue(retry_data['data']['resolved'])
        self.assertEqual(retry_data['data']['resolution_note'], 'Manual retry succeeded')

        mock_post.assert_called_once()

    def test_dead_letter_resolve_missing_note(self):
        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook'
            )
            db.session.add(subscription)
            db.session.commit()

            dead_letter = WebhookDeadLetter(
                subscription_id=subscription.id,
                event_type='test.event',
                event_data='{}',
                callback_url='https://example.com/webhook',
                last_status_code=500,
                retry_count=3
            )
            db.session.add(dead_letter)
            db.session.commit()
            dead_letter_id = dead_letter.id

        response = self.client.post(
            f'/api/dead-letters/{dead_letter_id}/resolve',
            data=json.dumps({'resolution_note': ''}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_get_retry_queue_stats(self):
        response = self.client.get('/api/retry-queue/stats')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('queue_size', data['data'])
        self.assertIn('max_retries', data['data'])
        self.assertIn('retry_intervals', data['data'])
        self.assertEqual(data['data']['max_retries'], 3)
        self.assertEqual(data['data']['retry_intervals'], [2, 5, 10])

    @patch('webhook_service.requests.post')
    def test_subscription_inactive_during_retry_moves_to_dead_letter(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=False
            )
            db.session.add(subscription)
            db.session.commit()

            retry_item = RetryItem(
                next_attempt_time=time.time(),
                priority=1,
                subscription_id=subscription.id,
                event_type='test.event',
                event_data={'user_id': 123},
                callback_url='https://example.com/webhook',
                secret_token=None,
                retry_count=1
            )

            webhook_service._execute_retry(retry_item)

            dead_letter = WebhookDeadLetter.query.filter_by(
                subscription_id=subscription.id
            ).first()

            self.assertIsNotNone(dead_letter)
            self.assertTrue(dead_letter.resolved)
            self.assertEqual(dead_letter.resolution_note, 'Subscription is inactive')

    @patch('webhook_service.requests.post')
    def test_subscription_deleted_during_retry_moves_to_dead_letter(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()
            subscription_id = subscription.id

            db.session.delete(subscription)
            db.session.commit()

            retry_item = RetryItem(
                next_attempt_time=time.time(),
                priority=1,
                subscription_id=subscription_id,
                event_type='test.event',
                event_data={'user_id': 123},
                callback_url='https://example.com/webhook',
                secret_token=None,
                retry_count=1
            )

            webhook_service._execute_retry(retry_item)

            dead_letter = WebhookDeadLetter.query.filter_by(
                subscription_id=subscription_id
            ).first()

            self.assertIsNotNone(dead_letter)
            self.assertTrue(dead_letter.resolved)
            self.assertEqual(dead_letter.resolution_note, 'Subscription no longer exists')

    @patch('webhook_service.requests.post')
    def test_connection_error_moves_to_retry_queue(self, mock_post):
        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError('Connection refused')

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            initial_queue_size = webhook_service.retry_queue.qsize()

            result = webhook_service._deliver_webhook(
                subscription,
                'test.event',
                {'user_id': 123}
            )

            self.assertIsNotNone(result.error_message)
            self.assertIn('Connection refused', result.error_message)
            self.assertEqual(webhook_service.retry_queue.qsize(), initial_queue_size + 1)

    def test_dead_letter_to_dict_includes_all_fields(self):
        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='test.event',
                callback_url='https://example.com/webhook'
            )
            db.session.add(subscription)
            db.session.commit()

            dead_letter = WebhookDeadLetter(
                subscription_id=subscription.id,
                event_type='test.event',
                event_data='{"key": "value"}',
                callback_url='https://example.com/webhook',
                secret_token='secret123',
                last_status_code=500,
                last_error_message='Internal Server Error',
                retry_count=3
            )
            db.session.add(dead_letter)
            db.session.commit()

            result = dead_letter.to_dict()

            self.assertEqual(result['event_type'], 'test.event')
            self.assertEqual(result['last_status_code'], 500)
            self.assertEqual(result['retry_count'], 3)
            self.assertEqual(result['last_error_message'], 'Internal Server Error')
            self.assertFalse(result['resolved'])
            self.assertIsNotNone(result['failed_at'])
            self.assertIsNone(result['resolved_at'])
            self.assertIsNone(result['resolution_note'])

    def test_signature_verification_valid(self):
        secret = 'my-secret-key'
        payload = '{"user_id": 123}'
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        signature = webhook_service.generate_signature(secret, payload, timestamp)

        with self.app.app_context():
            is_valid = webhook_service.verify_signature(
                secret_token=secret,
                payload=payload,
                timestamp=timestamp,
                signature=signature
            )
            self.assertTrue(is_valid)

    def test_signature_verification_wrong_secret(self):
        secret = 'my-secret-key'
        payload = '{"user_id": 123}'
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        signature = webhook_service.generate_signature(secret, payload, timestamp)

        with self.app.app_context():
            is_valid = webhook_service.verify_signature(
                secret_token='wrong-secret',
                payload=payload,
                timestamp=timestamp,
                signature=signature
            )
            self.assertFalse(is_valid)

    def test_signature_verification_expired_timestamp(self):
        secret = 'my-secret-key'
        payload = '{"user_id": 123}'
        expired_timestamp = str(int(datetime.now(timezone.utc).timestamp()) - 600)
        signature = webhook_service.generate_signature(secret, payload, expired_timestamp)

        with self.app.app_context():
            is_valid = webhook_service.verify_signature(
                secret_token=secret,
                payload=payload,
                timestamp=expired_timestamp,
                signature=signature
            )
            self.assertFalse(is_valid)

    def test_signature_verification_tampered_payload(self):
        secret = 'my-secret-key'
        payload = '{"user_id": 123}'
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        signature = webhook_service.generate_signature(secret, payload, timestamp)

        with self.app.app_context():
            is_valid = webhook_service.verify_signature(
                secret_token=secret,
                payload='{"user_id": 999}',
                timestamp=timestamp,
                signature=signature
            )
            self.assertFalse(is_valid)

    def test_signature_verification_api_endpoint(self):
        secret = 'my-secret-key'
        payload = '{"user_id": 123}'
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        signature = webhook_service.generate_signature(secret, payload, timestamp)

        valid_response = self.client.post(
            '/api/webhooks/verify-signature',
            data=json.dumps({
                'secret_token': secret,
                'payload': payload,
                'timestamp': timestamp,
                'signature': signature
            }),
            content_type='application/json'
        )
        self.assertEqual(valid_response.status_code, 200)
        valid_data = valid_response.get_json()
        self.assertTrue(valid_data['valid'])

        invalid_response = self.client.post(
            '/api/webhooks/verify-signature',
            data=json.dumps({
                'secret_token': 'wrong-secret',
                'payload': payload,
                'timestamp': timestamp,
                'signature': signature
            }),
            content_type='application/json'
        )
        self.assertEqual(invalid_response.status_code, 200)
        invalid_data = invalid_response.get_json()
        self.assertFalse(invalid_data['valid'])

    def test_signature_verification_api_missing_fields(self):
        response = self.client.post(
            '/api/webhooks/verify-signature',
            data=json.dumps({'secret_token': 'test'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    @patch('webhook_service.requests.post')
    def test_delivery_stats(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='user.created',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            for i in range(5):
                log = WebhookDeliveryLog(
                    subscription_id=subscription.id,
                    event_type='user.created',
                    event_data='{}',
                    callback_url='https://example.com/webhook',
                    status_code=200,
                    response_time_ms=100 + i * 10
                )
                db.session.add(log)

            log_fail = WebhookDeliveryLog(
                subscription_id=subscription.id,
                event_type='user.created',
                event_data='{}',
                callback_url='https://example.com/webhook',
                status_code=500,
                response_time_ms=200
            )
            db.session.add(log_fail)
            db.session.commit()

        stats_response = self.client.get('/api/stats')
        self.assertEqual(stats_response.status_code, 200)
        stats_data = stats_response.get_json()['data']

        self.assertEqual(stats_data['total_deliveries'], 6)
        self.assertEqual(stats_data['successful'], 5)
        self.assertEqual(stats_data['failed'], 1)
        self.assertAlmostEqual(stats_data['success_rate'], 83.33, places=1)
        self.assertIn('latency', stats_data)
        self.assertIn('avg_ms', stats_data['latency'])
        self.assertIn('min_ms', stats_data['latency'])
        self.assertIn('max_ms', stats_data['latency'])
        self.assertIn('p50_ms', stats_data['latency'])
        self.assertEqual(stats_data['latency']['min_ms'], 100)
        self.assertEqual(stats_data['latency']['max_ms'], 200)

    @patch('webhook_service.requests.post')
    def test_subscription_stats_endpoint(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        create_response = self.client.post(
            '/api/webhooks',
            data=json.dumps({
                'event_type': 'user.created',
                'callback_url': 'https://example.com/webhook'
            }),
            content_type='application/json'
        )
        webhook_id = create_response.get_json()['data']['id']

        self.client.post(
            '/api/events/user.created',
            data=json.dumps({'data': {'user_id': 1}, 'wait_for_completion': True}),
            content_type='application/json'
        )

        stats_response = self.client.get(f'/api/webhooks/{webhook_id}/stats')
        self.assertEqual(stats_response.status_code, 200)
        stats_data = stats_response.get_json()['data']
        self.assertGreaterEqual(stats_data['total_deliveries'], 1)
        self.assertIn('success_rate', stats_data)
        self.assertIn('latency', stats_data)

    def test_stats_endpoint_not_found_subscription(self):
        response = self.client.get('/api/webhooks/999/stats')
        self.assertEqual(response.status_code, 404)

    @patch('webhook_service.requests.post')
    def test_stats_with_latency_recording(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'

        def mock_post_fn(url, data, headers, timeout):
            time.sleep(0.05)
            return mock_response

        mock_post.side_effect = mock_post_fn

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='latency.test',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            result = webhook_service._deliver_webhook_internal(
                subscription, 'latency.test', {'key': 'value'}
            )

            self.assertIsNotNone(result.response_time_ms)
            self.assertGreater(result.response_time_ms, 0)

    @patch('webhook_service.requests.post')
    def test_replay_single_event(self, mock_post):
        mock_first = MagicMock()
        mock_first.status_code = 200
        mock_first.text = 'OK'

        mock_replay = MagicMock()
        mock_replay.status_code = 200
        mock_replay.text = 'Replayed OK'

        mock_post.side_effect = [mock_first, mock_replay]

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='user.created',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            original_log = webhook_service._deliver_webhook_internal(
                subscription, 'user.created', {'user_id': 42, 'action': 'signup'}
            )
            original_log_id = original_log.id

        replay_response = self.client.post(f'/api/logs/{original_log_id}/replay')
        self.assertEqual(replay_response.status_code, 200)
        replay_data = replay_response.get_json()
        self.assertEqual(replay_data['data']['original_log_id'], original_log_id)
        self.assertIn('new_log', replay_data['data'])

    @patch('webhook_service.requests.post')
    def test_replay_single_event_not_found(self, mock_post):
        response = self.client.post('/api/logs/99999/replay')
        self.assertEqual(response.status_code, 404)

    @patch('webhook_service.requests.post')
    def test_replay_single_event_inactive_subscription(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='user.created',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            log = webhook_service._deliver_webhook_internal(
                subscription, 'user.created', {'user_id': 1}
            )
            log_id = log.id

            subscription.is_active = False
            db.session.commit()

        response = self.client.post(f'/api/logs/{log_id}/replay')
        self.assertEqual(response.status_code, 404)

    @patch('webhook_service.requests.post')
    def test_replay_events_by_type(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='order.paid',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            for i in range(3):
                webhook_service._deliver_webhook_internal(
                    subscription, 'order.paid', {'order_id': i}
                )

            while not webhook_service.retry_queue.empty():
                webhook_service.retry_queue.get()

        replay_response = self.client.post(
            '/api/events/order.paid/replay',
            data=json.dumps({'limit': 10}),
            content_type='application/json'
        )
        self.assertEqual(replay_response.status_code, 200)
        replay_data = replay_response.get_json()
        self.assertEqual(replay_data['replayed_count'], 3)

    @patch('webhook_service.requests.post')
    def test_replay_events_by_type_with_limit(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='order.paid',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            for i in range(5):
                webhook_service._deliver_webhook_internal(
                    subscription, 'order.paid', {'order_id': i}
                )

            while not webhook_service.retry_queue.empty():
                webhook_service.retry_queue.get()

        replay_response = self.client.post(
            '/api/events/order.paid/replay',
            data=json.dumps({'limit': 2}),
            content_type='application/json'
        )
        self.assertEqual(replay_response.status_code, 200)
        replay_data = replay_response.get_json()
        self.assertEqual(replay_data['replayed_count'], 2)

    @patch('webhook_service.requests.post')
    def test_event_type_stats_endpoint(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_post.return_value = mock_response

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='stats.test',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            webhook_service._deliver_webhook_internal(
                subscription, 'stats.test', {'key': 'value'}
            )

        stats_response = self.client.get('/api/events/stats.test/stats')
        self.assertEqual(stats_response.status_code, 200)
        stats_data = stats_response.get_json()['data']
        self.assertGreaterEqual(stats_data['total_deliveries'], 1)

    @patch('webhook_service.requests.post')
    def test_delivery_log_includes_response_time(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'OK'

        def mock_post_fn(url, data, headers, timeout):
            time.sleep(0.01)
            return mock_response

        mock_post.side_effect = mock_post_fn

        with self.app.app_context():
            subscription = WebhookSubscription(
                event_type='latency.test',
                callback_url='https://example.com/webhook',
                is_active=True
            )
            db.session.add(subscription)
            db.session.commit()

            result = webhook_service._deliver_webhook_internal(
                subscription, 'latency.test', {'key': 'value'}
            )

            self.assertIsNotNone(result.response_time_ms)
            self.assertGreater(result.response_time_ms, 0)

            log_dict = result.to_dict()
            self.assertIn('response_time_ms', log_dict)
            self.assertIsNotNone(log_dict['response_time_ms'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
