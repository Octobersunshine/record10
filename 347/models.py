from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class IPList(db.Model):
    __tablename__ = 'ip_list'
    __table_args__ = (
        db.UniqueConstraint('ip_address', 'list_type', name='uq_ip_list_type'),
    )

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), nullable=False, index=True)
    list_type = db.Column(db.String(10), nullable=False, index=True)
    is_cidr = db.Column(db.Boolean, default=False)
    description = db.Column(db.String(255), default='')
    expires_at = db.Column(db.DateTime, nullable=True, index=True)
    is_temporary = db.Column(db.Boolean, default=False)
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
            'is_temporary': self.is_temporary,
            'is_expired': self.is_expired(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class BanAttempt(db.Model):
    __tablename__ = 'ban_attempts'

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), nullable=False, index=True)
    failure_count = db.Column(db.Integer, default=0)
    last_attempt_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    first_attempt_at = db.Column(db.DateTime, default=datetime.utcnow)
    window_start = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'ip_address': self.ip_address,
            'failure_count': self.failure_count,
            'last_attempt_at': self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            'first_attempt_at': self.first_attempt_at.isoformat() if self.first_attempt_at else None,
            'window_start': self.window_start.isoformat() if self.window_start else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class BanLog(db.Model):
    __tablename__ = 'ban_logs'

    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(50), nullable=False, index=True)
    event_type = db.Column(db.String(20), nullable=False, index=True)
    reason = db.Column(db.String(255), default='')
    source = db.Column(db.String(50), default='')
    ban_duration_seconds = db.Column(db.Integer, nullable=True)
    threshold = db.Column(db.Integer, nullable=True)
    failure_count = db.Column(db.Integer, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id,
            'ip_address': self.ip_address,
            'event_type': self.event_type,
            'reason': self.reason,
            'source': self.source,
            'ban_duration_seconds': self.ban_duration_seconds,
            'threshold': self.threshold,
            'failure_count': self.failure_count,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
