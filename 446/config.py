class Config:
    AUTO_BAN_ENABLED = True
    MAX_FAILURE_ATTEMPTS = 5
    FAILURE_WINDOW_SECONDS = 300
    AUTO_BAN_DURATION_SECONDS = 3600
    PERMANENT_BAN_AFTER = 3
    CLEANUP_INTERVAL_SECONDS = 60
    ENABLE_NOTIFICATIONS = True

    @classmethod
    def get_config(cls):
        return {
            'auto_ban_enabled': cls.AUTO_BAN_ENABLED,
            'max_failure_attempts': cls.MAX_FAILURE_ATTEMPTS,
            'failure_window_seconds': cls.FAILURE_WINDOW_SECONDS,
            'auto_ban_duration_seconds': cls.AUTO_BAN_DURATION_SECONDS,
            'permanent_ban_after': cls.PERMANENT_BAN_AFTER,
            'cleanup_interval_seconds': cls.CLEANUP_INTERVAL_SECONDS,
            'enable_notifications': cls.ENABLE_NOTIFICATIONS
        }

    @classmethod
    def update_config(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
        return cls.get_config()
