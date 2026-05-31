import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.example.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', 'your_email@example.com')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', 'your_password')
    SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'True').lower() == 'true'
    SMTP_FROM_EMAIL = os.getenv('SMTP_FROM_EMAIL', 'your_email@example.com')
    SMTP_FROM_NAME = os.getenv('SMTP_FROM_NAME', 'Email Sender')
    SMTP_TIMEOUT = int(os.getenv('SMTP_TIMEOUT', 300))
    SMTP_MAX_RETRIES = int(os.getenv('SMTP_MAX_RETRIES', 3))
    SMTP_RETRY_DELAY = int(os.getenv('SMTP_RETRY_DELAY', 5))
    SMTP_LARGE_ATTACHMENT_THRESHOLD = int(os.getenv('SMTP_LARGE_ATTACHMENT_THRESHOLD', 10 * 1024 * 1024))
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'
