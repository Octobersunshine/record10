from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class WebhookSubscription(db.Model):
    __tablename__ = 'webhook_subscriptions'

    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(100), nullable=False, index=True)
    callback_url = db.Column(db.String(500), nullable=False)
    description = db.Column(db.String(255))
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    secret_token = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    __table_args__ = (
        db.UniqueConstraint('event_type', 'callback_url', name='uq_event_callback'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'event_type': self.event_type,
            'callback_url': self.callback_url,
            'description': self.description,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class WebhookDeliveryLog(db.Model):
    __tablename__ = 'webhook_delivery_logs'

    id = db.Column(db.Integer, primary_key=True)
    subscription_id = db.Column(
        db.Integer,
        db.ForeignKey('webhook_subscriptions.id', ondelete='CASCADE'),
        nullable=False
    )
    event_type = db.Column(db.String(100), nullable=False, index=True)
    event_data = db.Column(db.Text, nullable=False)
    callback_url = db.Column(db.String(500), nullable=False)
    status_code = db.Column(db.Integer)
    response_body = db.Column(db.Text)
    error_message = db.Column(db.String(500))
    delivered_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    retry_count = db.Column(db.Integer, default=0, nullable=False)
    response_time_ms = db.Column(db.Integer)

    subscription = db.relationship(
        'WebhookSubscription',
        backref=db.backref('delivery_logs', cascade='all, delete-orphan')
    )

    def to_dict(self):
        return {
            'id': self.id,
            'subscription_id': self.subscription_id,
            'event_type': self.event_type,
            'callback_url': self.callback_url,
            'status_code': self.status_code,
            'error_message': self.error_message,
            'delivered_at': self.delivered_at.isoformat(),
            'retry_count': self.retry_count,
            'response_time_ms': self.response_time_ms
        }


class WebhookDeadLetter(db.Model):
    __tablename__ = 'webhook_dead_letters'

    id = db.Column(db.Integer, primary_key=True)
    subscription_id = db.Column(
        db.Integer,
        db.ForeignKey('webhook_subscriptions.id', ondelete='SET NULL'),
        nullable=True
    )
    event_type = db.Column(db.String(100), nullable=False, index=True)
    event_data = db.Column(db.Text, nullable=False)
    callback_url = db.Column(db.String(500), nullable=False)
    secret_token = db.Column(db.String(100))
    last_status_code = db.Column(db.Integer)
    last_error_message = db.Column(db.String(500))
    retry_count = db.Column(db.Integer, default=0, nullable=False)
    failed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    resolved_at = db.Column(db.DateTime)
    resolved = db.Column(db.Boolean, default=False, nullable=False)
    resolution_note = db.Column(db.String(500))

    subscription = db.relationship(
        'WebhookSubscription',
        backref=db.backref('dead_letters')
    )

    def to_dict(self):
        return {
            'id': self.id,
            'subscription_id': self.subscription_id,
            'event_type': self.event_type,
            'callback_url': self.callback_url,
            'last_status_code': self.last_status_code,
            'last_error_message': self.last_error_message,
            'retry_count': self.retry_count,
            'failed_at': self.failed_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved': self.resolved,
            'resolution_note': self.resolution_note
        }
