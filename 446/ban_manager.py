from datetime import datetime, timedelta
from models import db, IPList, FailureAttempt, BanStats, Notification
from config import Config
from ip_utils import is_valid_ip_or_cidr


class BanManager:
    notification_callbacks = []

    @classmethod
    def register_notification_callback(cls, callback):
        cls.notification_callbacks.append(callback)

    @classmethod
    def _send_notification(cls, notification_type, ip_address, message):
        if not Config.ENABLE_NOTIFICATIONS:
            return

        notification = Notification(
            notification_type=notification_type,
            ip_address=ip_address,
            message=message
        )
        db.session.add(notification)
        db.session.commit()

        for callback in cls.notification_callbacks:
            try:
                callback({
                    'type': notification_type,
                    'ip': ip_address,
                    'message': message,
                    'time': datetime.utcnow().isoformat()
                })
            except Exception as e:
                print(f"Notification callback error: {e}")

        print(f"[NOTIFICATION] {notification_type}: {message}")

    @classmethod
    def record_failure(cls, ip_address, failure_reason=None, request_path=None, user_agent=None):
        if not Config.AUTO_BAN_ENABLED:
            return None

        valid, _ = is_valid_ip_or_cidr(ip_address)
        if not valid:
            return None

        attempt = FailureAttempt(
            ip_address=ip_address,
            failure_reason=failure_reason,
            request_path=request_path,
            user_agent=user_agent
        )
        db.session.add(attempt)
        db.session.commit()

        return cls._check_and_ban(ip_address)

    @classmethod
    def _check_and_ban(cls, ip_address):
        window_start = datetime.utcnow() - timedelta(seconds=Config.FAILURE_WINDOW_SECONDS)

        failure_count = FailureAttempt.query.filter(
            FailureAttempt.ip_address == ip_address,
            FailureAttempt.attempt_time >= window_start
        ).count()

        if failure_count >= Config.MAX_FAILURE_ATTEMPTS:
            existing_ban = IPList.query.filter_by(
                ip_address=ip_address,
                list_type='blacklist'
            ).first()

            if existing_ban and not existing_ban.is_expired():
                return None

            ban_count = BanStats.query.filter_by(
                ip_address=ip_address,
                is_automatic=True
            ).count()

            is_permanent = ban_count >= Config.PERMANENT_BAN_AFTER

            if is_permanent:
                expires_at = None
                duration = None
                reason = f"自动永久封禁：{failure_count}次失败尝试，累计{ban_count + 1}次封禁"
            else:
                expires_at = datetime.utcnow() + timedelta(seconds=Config.AUTO_BAN_DURATION_SECONDS)
                duration = Config.AUTO_BAN_DURATION_SECONDS
                reason = f"自动临时封禁：{failure_count}次失败尝试（{Config.AUTO_BAN_DURATION_SECONDS}秒）"

            if existing_ban:
                existing_ban.expires_at = expires_at
                existing_ban.is_autoban = True
                existing_ban.description = reason
                existing_ban.updated_at = datetime.utcnow()
            else:
                new_ban = IPList(
                    ip_address=ip_address,
                    list_type='blacklist',
                    is_cidr=False,
                    description=reason,
                    expires_at=expires_at,
                    is_autoban=True
                )
                db.session.add(new_ban)

            ban_stat = BanStats(
                ip_address=ip_address,
                failure_count=failure_count,
                ban_duration_seconds=duration,
                is_automatic=True,
                is_active=True,
                reason=reason
            )
            db.session.add(ban_stat)
            db.session.commit()

            cls._send_notification(
                'auto_ban',
                ip_address,
                f"IP {ip_address} 已被{'永久' if is_permanent else '临时'}封禁，失败次数: {failure_count}"
            )

            FailureAttempt.query.filter_by(ip_address=ip_address).delete()
            db.session.commit()

            return {
                'ip_address': ip_address,
                'banned': True,
                'is_permanent': is_permanent,
                'expires_at': expires_at.isoformat() if expires_at else None,
                'failure_count': failure_count,
                'reason': reason
            }

        return None

    @classmethod
    def cleanup_expired_bans(cls):
        now = datetime.utcnow()
        expired_bans = IPList.query.filter(
            IPList.list_type == 'blacklist',
            IPList.expires_at.isnot(None),
            IPList.expires_at <= now
        ).all()

        cleaned_count = 0
        for ban in expired_bans:
            ban_stat = BanStats.query.filter_by(
                ip_address=ban.ip_address,
                is_active=True
            ).first()
            if ban_stat:
                ban_stat.is_active = False
                ban_stat.unban_time = now

            db.session.delete(ban)
            cleaned_count += 1

            cls._send_notification(
                'auto_unban',
                ban.ip_address,
                f"IP {ban.ip_address} 临时封禁已到期，自动解封"
            )

        if cleaned_count > 0:
            db.session.commit()

        return cleaned_count

    @classmethod
    def get_failure_count(cls, ip_address):
        window_start = datetime.utcnow() - timedelta(seconds=Config.FAILURE_WINDOW_SECONDS)
        return FailureAttempt.query.filter(
            FailureAttempt.ip_address == ip_address,
            FailureAttempt.attempt_time >= window_start
        ).count()

    @classmethod
    def get_ban_stats(cls, ip_address=None, limit=100):
        query = BanStats.query
        if ip_address:
            query = query.filter_by(ip_address=ip_address)
        return query.order_by(BanStats.ban_time.desc()).limit(limit).all()

    @classmethod
    def get_active_bans(cls):
        now = datetime.utcnow()
        return IPList.query.filter(
            IPList.list_type == 'blacklist',
            (IPList.expires_at.is_(None)) | (IPList.expires_at > now)
        ).all()

    @classmethod
    def get_notifications(cls, unread_only=False, limit=100):
        query = Notification.query
        if unread_only:
            query = query.filter_by(is_read=False)
        return query.order_by(Notification.created_at.desc()).limit(limit).all()

    @classmethod
    def mark_notification_read(cls, notification_id):
        notification = Notification.query.get(notification_id)
        if notification:
            notification.is_read = True
            db.session.commit()
            return True
        return False

    @classmethod
    def mark_all_notifications_read(cls):
        Notification.query.update({Notification.is_read: True})
        db.session.commit()

    @classmethod
    def get_ban_summary(cls):
        now = datetime.utcnow()
        total_bans = IPList.query.filter_by(list_type='blacklist').count()
        active_bans = IPList.query.filter(
            IPList.list_type == 'blacklist',
            (IPList.expires_at.is_(None)) | (IPList.expires_at > now)
        ).count()
        expired_bans = total_bans - active_bans
        temporary_bans = IPList.query.filter(
            IPList.list_type == 'blacklist',
            IPList.expires_at.isnot(None)
        ).count()
        permanent_bans = IPList.query.filter(
            IPList.list_type == 'blacklist',
            IPList.expires_at.is_(None)
        ).count()
        auto_bans = IPList.query.filter_by(
            list_type='blacklist',
            is_autoban=True
        ).count()
        manual_bans = total_bans - auto_bans

        total_ban_stats = BanStats.query.count()
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        bans_today = BanStats.query.filter(BanStats.ban_time >= today_start).count()
        unread_notifications = Notification.query.filter_by(is_read=False).count()

        return {
            'total_bans': total_bans,
            'active_bans': active_bans,
            'expired_bans': expired_bans,
            'temporary_bans': temporary_bans,
            'permanent_bans': permanent_bans,
            'auto_bans': auto_bans,
            'manual_bans': manual_bans,
            'total_ban_events': total_ban_stats,
            'bans_today': bans_today,
            'unread_notifications': unread_notifications
        }
