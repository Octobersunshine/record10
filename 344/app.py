import os
import json
import logging
from urllib.parse import urlparse

from flask import Flask, jsonify, request

from models import db
from webhook_service import WebhookService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

webhook_service = WebhookService()


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False


def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL', 'sqlite:///webhooks.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['WEBHOOK_MAX_WORKERS'] = int(
        os.environ.get('WEBHOOK_MAX_WORKERS', 10)
    )

    db.init_app(app)
    webhook_service.init_app(app)

    with app.app_context():
        db.create_all()
        logger.info("Database tables initialized")

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'ok',
            'service': 'webhook-api'
        })

    @app.route('/api/webhooks', methods=['POST'])
    def create_webhook():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        event_type = data.get('event_type')
        callback_url = data.get('callback_url')

        if not event_type or not callback_url:
            return jsonify({
                'error': 'event_type and callback_url are required'
            }), 400

        if not isinstance(event_type, str) or len(event_type.strip()) == 0:
            return jsonify({'error': 'event_type must be a non-empty string'}), 400

        if not is_valid_url(callback_url):
            return jsonify({'error': 'callback_url must be a valid HTTP/HTTPS URL'}), 400

        description = data.get('description')
        secret_token = data.get('secret_token')
        is_active = data.get('is_active', True)

        if not isinstance(is_active, bool):
            return jsonify({'error': 'is_active must be a boolean'}), 400

        try:
            subscription = webhook_service.create_subscription(
                event_type=event_type.strip(),
                callback_url=callback_url,
                description=description,
                secret_token=secret_token,
                is_active=is_active
            )
            return jsonify({
                'message': 'Webhook subscription created successfully',
                'data': subscription.to_dict()
            }), 201
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")
            return jsonify({'error': 'Failed to create webhook subscription'}), 500

    @app.route('/api/webhooks', methods=['GET'])
    def list_webhooks():
        event_type = request.args.get('event_type')
        is_active_param = request.args.get('is_active')

        is_active = None
        if is_active_param is not None:
            if is_active_param.lower() == 'true':
                is_active = True
            elif is_active_param.lower() == 'false':
                is_active = False
            else:
                return jsonify({'error': 'is_active must be true or false'}), 400

        subscriptions = webhook_service.list_subscriptions(
            event_type=event_type,
            is_active=is_active
        )

        return jsonify({
            'data': [s.to_dict() for s in subscriptions],
            'count': len(subscriptions)
        })

    @app.route('/api/webhooks/<int:subscription_id>', methods=['GET'])
    def get_webhook(subscription_id):
        subscription = webhook_service.get_subscription(subscription_id)
        if not subscription:
            return jsonify({'error': 'Webhook subscription not found'}), 404

        return jsonify({'data': subscription.to_dict()})

    @app.route('/api/webhooks/<int:subscription_id>', methods=['PUT'])
    def update_webhook(subscription_id):
        subscription = webhook_service.get_subscription(subscription_id)
        if not subscription:
            return jsonify({'error': 'Webhook subscription not found'}), 404

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        update_data = {}

        if 'event_type' in data:
            event_type = data['event_type']
            if not isinstance(event_type, str) or len(event_type.strip()) == 0:
                return jsonify({'error': 'event_type must be a non-empty string'}), 400
            update_data['event_type'] = event_type.strip()

        if 'callback_url' in data:
            callback_url = data['callback_url']
            if not is_valid_url(callback_url):
                return jsonify({'error': 'callback_url must be a valid HTTP/HTTPS URL'}), 400
            update_data['callback_url'] = callback_url

        if 'description' in data:
            update_data['description'] = data['description']

        if 'secret_token' in data:
            update_data['secret_token'] = data['secret_token']

        if 'is_active' in data:
            if not isinstance(data['is_active'], bool):
                return jsonify({'error': 'is_active must be a boolean'}), 400
            update_data['is_active'] = data['is_active']

        if not update_data:
            return jsonify({'error': 'No valid fields to update'}), 400

        try:
            updated = webhook_service.update_subscription(
                subscription_id, **update_data
            )
            return jsonify({
                'message': 'Webhook subscription updated successfully',
                'data': updated.to_dict()
            })
        except Exception as e:
            logger.error(f"Error updating webhook {subscription_id}: {e}")
            return jsonify({'error': 'Failed to update webhook subscription'}), 500

    @app.route('/api/webhooks/<int:subscription_id>', methods=['DELETE'])
    def delete_webhook(subscription_id):
        deleted = webhook_service.delete_subscription(subscription_id)
        if not deleted:
            return jsonify({'error': 'Webhook subscription not found'}), 404

        return jsonify({
            'message': 'Webhook subscription deleted successfully'
        })

    @app.route('/api/events', methods=['GET'])
    def list_event_types():
        event_types = webhook_service.get_event_types()
        return jsonify({
            'data': event_types,
            'count': len(event_types)
        })

    @app.route('/api/events/<event_type>', methods=['POST'])
    def trigger_event(event_type):
        if not event_type or len(event_type.strip()) == 0:
            return jsonify({'error': 'event_type is required'}), 400

        data = request.get_json() or {}
        event_data = data.get('data', data)
        wait_for_completion = data.get('wait_for_completion', False)

        if not isinstance(event_data, dict):
            return jsonify({'error': 'event data must be a JSON object'}), 400

        subscribers = webhook_service.get_active_subscribers(event_type)
        if not subscribers:
            return jsonify({
                'message': f'Event {event_type} triggered, but no active subscribers',
                'subscriber_count': 0,
                'deliveries': []
            })

        try:
            deliveries = webhook_service.trigger_event(
                event_type=event_type,
                event_data=event_data,
                wait_for_completion=wait_for_completion
            )

            successful = sum(
                1 for d in deliveries if d.status_code and 200 <= d.status_code < 300
            )
            failed = len(deliveries) - successful

            return jsonify({
                'message': f'Event {event_type} triggered successfully',
                'subscriber_count': len(subscribers),
                'successful_deliveries': successful,
                'failed_deliveries': failed,
                'deliveries': [d.to_dict() for d in deliveries]
            })
        except Exception as e:
            logger.error(f"Error triggering event {event_type}: {e}")
            return jsonify({'error': 'Failed to trigger event'}), 500

    @app.route('/api/webhooks/<int:subscription_id>/logs', methods=['GET'])
    def get_webhook_logs(subscription_id):
        subscription = webhook_service.get_subscription(subscription_id)
        if not subscription:
            return jsonify({'error': 'Webhook subscription not found'}), 404

        try:
            limit = int(request.args.get('limit', 100))
            if limit < 1 or limit > 1000:
                return jsonify({'error': 'limit must be between 1 and 1000'}), 400
        except ValueError:
            return jsonify({'error': 'limit must be an integer'}), 400

        logs = webhook_service.get_delivery_logs(
            subscription_id=subscription_id,
            limit=limit
        )

        return jsonify({
            'data': [l.to_dict() for l in logs],
            'count': len(logs)
        })

    @app.route('/api/logs', methods=['GET'])
    def get_all_logs():
        event_type = request.args.get('event_type')

        try:
            limit = int(request.args.get('limit', 100))
            if limit < 1 or limit > 1000:
                return jsonify({'error': 'limit must be between 1 and 1000'}), 400
        except ValueError:
            return jsonify({'error': 'limit must be an integer'}), 400

        logs = webhook_service.get_delivery_logs(
            event_type=event_type,
            limit=limit
        )

        return jsonify({
            'data': [l.to_dict() for l in logs],
            'count': len(logs)
        })

    @app.route('/api/dead-letters', methods=['GET'])
    def list_dead_letters():
        resolved_param = request.args.get('resolved')
        event_type = request.args.get('event_type')

        resolved = None
        if resolved_param is not None:
            if resolved_param.lower() == 'true':
                resolved = True
            elif resolved_param.lower() == 'false':
                resolved = False
            else:
                return jsonify({'error': 'resolved must be true or false'}), 400

        try:
            limit = int(request.args.get('limit', 100))
            if limit < 1 or limit > 1000:
                return jsonify({'error': 'limit must be between 1 and 1000'}), 400
        except ValueError:
            return jsonify({'error': 'limit must be an integer'}), 400

        dead_letters = webhook_service.get_dead_letters(
            resolved=resolved,
            event_type=event_type,
            limit=limit
        )

        return jsonify({
            'data': [d.to_dict() for d in dead_letters],
            'count': len(dead_letters)
        })

    @app.route('/api/dead-letters/<int:dead_letter_id>', methods=['GET'])
    def get_dead_letter(dead_letter_id):
        dead_letter = webhook_service.get_dead_letter(dead_letter_id)
        if not dead_letter:
            return jsonify({'error': 'Dead letter not found'}), 404

        result = dead_letter.to_dict()
        try:
            result['event_data'] = json.loads(dead_letter.event_data)
        except json.JSONDecodeError:
            pass

        return jsonify({'data': result})

    @app.route('/api/dead-letters/<int:dead_letter_id>/retry', methods=['POST'])
    def retry_dead_letter(dead_letter_id):
        dead_letter = webhook_service.retry_dead_letter(dead_letter_id)
        if not dead_letter:
            return jsonify({'error': 'Dead letter not found'}), 404

        return jsonify({
            'message': 'Dead letter retry initiated',
            'data': dead_letter.to_dict()
        })

    @app.route('/api/dead-letters/<int:dead_letter_id>/resolve', methods=['POST'])
    def resolve_dead_letter(dead_letter_id):
        data = request.get_json() or {}
        resolution_note = data.get('resolution_note', '')

        if not isinstance(resolution_note, str):
            return jsonify({'error': 'resolution_note must be a string'}), 400

        if len(resolution_note.strip()) == 0:
            return jsonify({'error': 'resolution_note is required'}), 400

        dead_letter = webhook_service.resolve_dead_letter(
            dead_letter_id, resolution_note
        )
        if not dead_letter:
            return jsonify({'error': 'Dead letter not found'}), 404

        return jsonify({
            'message': 'Dead letter marked as resolved',
            'data': dead_letter.to_dict()
        })

    @app.route('/api/dead-letters/<int:dead_letter_id>', methods=['DELETE'])
    def delete_dead_letter(dead_letter_id):
        deleted = webhook_service.delete_dead_letter(dead_letter_id)
        if not deleted:
            return jsonify({'error': 'Dead letter not found'}), 404

        return jsonify({
            'message': 'Dead letter deleted successfully'
        })

    @app.route('/api/retry-queue/stats', methods=['GET'])
    def get_retry_queue_stats():
        stats = webhook_service.get_retry_queue_stats()
        return jsonify({'data': stats})

    @app.route('/api/webhooks/verify-signature', methods=['POST'])
    def verify_webhook_signature():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        required_fields = ['secret_token', 'payload', 'timestamp', 'signature']
        for field_name in required_fields:
            if field_name not in data:
                return jsonify({'error': f'{field_name} is required'}), 400

        tolerance_seconds = data.get('tolerance_seconds', 300)
        if not isinstance(tolerance_seconds, int) or tolerance_seconds < 0:
            return jsonify({'error': 'tolerance_seconds must be a non-negative integer'}), 400

        is_valid = webhook_service.verify_signature(
            secret_token=data['secret_token'],
            payload=data['payload'],
            timestamp=data['timestamp'],
            signature=data['signature'],
            tolerance_seconds=tolerance_seconds
        )

        return jsonify({
            'valid': is_valid,
            'message': 'Signature is valid' if is_valid else 'Signature verification failed'
        })

    @app.route('/api/stats', methods=['GET'])
    def get_delivery_stats():
        subscription_id = request.args.get('subscription_id')
        event_type = request.args.get('event_type')

        sub_id = None
        if subscription_id:
            try:
                sub_id = int(subscription_id)
            except ValueError:
                return jsonify({'error': 'subscription_id must be an integer'}), 400

        stats = webhook_service.get_delivery_stats(
            subscription_id=sub_id,
            event_type=event_type
        )

        return jsonify({'data': stats})

    @app.route('/api/webhooks/<int:subscription_id>/stats', methods=['GET'])
    def get_subscription_stats(subscription_id):
        subscription = webhook_service.get_subscription(subscription_id)
        if not subscription:
            return jsonify({'error': 'Webhook subscription not found'}), 404

        stats = webhook_service.get_delivery_stats(subscription_id=subscription_id)
        return jsonify({'data': stats})

    @app.route('/api/events/<event_type>/stats', methods=['GET'])
    def get_event_type_stats(event_type):
        stats = webhook_service.get_delivery_stats(event_type=event_type)
        return jsonify({'data': stats})

    @app.route('/api/logs/<int:log_id>/replay', methods=['POST'])
    def replay_single_event(log_id):
        new_log = webhook_service.replay_event(log_id)
        if not new_log:
            return jsonify({'error': 'Cannot replay event: log not found, subscription inactive, or invalid data'}), 404

        return jsonify({
            'message': 'Event replayed successfully',
            'data': {
                'original_log_id': log_id,
                'new_log': new_log.to_dict()
            }
        })

    @app.route('/api/events/<event_type>/replay', methods=['POST'])
    def replay_events_by_type(event_type):
        data = request.get_json() or {}

        try:
            limit = int(data.get('limit', 100))
            if limit < 1 or limit > 1000:
                return jsonify({'error': 'limit must be between 1 and 1000'}), 400
        except ValueError:
            return jsonify({'error': 'limit must be an integer'}), 400

        results = webhook_service.replay_events_by_type(event_type, limit=limit)

        return jsonify({
            'message': f'Replayed {len(results)} events for {event_type}',
            'replayed_count': len(results),
            'data': [r.to_dict() for r in results]
        })

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({'error': 'Method not allowed'}), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

    return app


app = create_app()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
