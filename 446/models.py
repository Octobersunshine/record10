from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class IPList(db.Model):
    __tablename__ = 'ip_list'

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), nullable=False)
    __table_args__ = (
        db.UniqueConstraint('ip_address', 'list_type', name='_ip_listtype_uc'),
    )
    list_type = db.Column(db.Enum('whitelist', 'blacklist'), nullable=False)
    is_cidr = db.Column(db.Boolean, nullable=False, default=False)
    description = db.Column(db.String(255))
    expires_at = db.Column(db.DateTime, nullable=True)
    is_autoban = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def is_expired(self):
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self):
        return {
            'id': self.id,
            'ip_address': self.ip_address,
            'list_type': self.list_type,
            'is_cidr': self.is_cidr,
            'description': self.description,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_autoban': self.is_autoban,
            'is_expired': self.is_expired(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class FailureAttempt(db.Model):
    __tablename__ = 'failure_attempts'

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), nullable=False, index=True)
    attempt_time = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    failure_reason = db.Column(db.String(255))
    request_path = db.Column(db.String(255))
    user_agent = db.Column(db.String(500))

    def to_dict(self):
        return {
            'id': self.id,
            'ip_address': self.ip_address,
            'attempt_time': self.attempt_time.isoformat() if self.attempt_time else None,
            'failure_reason': self.failure_reason,
            'request_path': self.request_path,
            'user_agent': self.user_agent
        }


class BanStats(db.Model):
    __tablename__ = 'ban_stats'

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), nullable=False)
    ban_time = db.Column(db.DateTime, default=datetime.utcnow)
    unban_time = db.Column(db.DateTime, nullable=True)
    failure_count = db.Column(db.Integer, nullable=False, default=0)
    ban_duration_seconds = db.Column(db.Integer, nullable=True)
    is_automatic = db.Column(db.Boolean, nullable=False, default=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    reason = db.Column(db.String(255))

    def to_dict(self):
        return {
            'id': self.id,
            'ip_address': self.ip_address,
            'ban_time': self.ban_time.isoformat() if self.ban_time else None,
            'unban_time': self.unban_time.isoformat() if self.unban_time else None,
            'failure_count': self.failure_count,
            'ban_duration_seconds': self.ban_duration_seconds,
            'is_automatic': self.is_automatic,
            'is_active': self.is_active,
            'reason': self.reason
        }


class Notification(db.Model):
    __tablename__ = 'notifications'

    id = db.Column(db.Integer, primary_key=True)
    notification_type = db.Column(db.String(50), nullable=False)
    ip_address = db.Column(db.String(50))
    message = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, nullable=False, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'notification_type': self.notification_type,
            'ip_address': self.ip_address,
            'message': self.message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_read': self.is_read
        }
