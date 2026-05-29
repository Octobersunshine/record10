import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmailConfig:
    smtp_server: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    use_tls: bool = True
    sender_name: str = ""
    sender_email: str = ""

    @classmethod
    def from_env(cls) -> "EmailConfig":
        return cls(
            smtp_server=os.getenv("SMTP_SERVER", ""),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            use_tls=os.getenv("SMTP_USE_TLS", "True").lower() == "true",
            sender_name=os.getenv("SENDER_NAME", ""),
            sender_email=os.getenv("SENDER_EMAIL", os.getenv("SMTP_USERNAME", "")),
        )

    def validate(self) -> None:
        required_fields = [
            ("smtp_server", self.smtp_server),
            ("smtp_port", self.smtp_port),
            ("smtp_username", self.smtp_username),
            ("smtp_password", self.smtp_password),
            ("sender_email", self.sender_email),
        ]
        missing = [name for name, value in required_fields if not value]
        if missing:
            raise ValueError(f"Missing required email config fields: {', '.join(missing)}")
