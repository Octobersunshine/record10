import smtplib
import logging
import os
import time
import mimetypes
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr, formatdate, make_msgid
from typing import List, Optional, Union, Dict, Any

from config import EmailConfig

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20
DEFAULT_BATCH_DELAY = 1.0
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024


@dataclass
class AttachmentInfo:
    filename: str
    size: int
    content_type: str


@dataclass
class BatchResult:
    batch_index: int
    recipients: List[str]
    success: bool
    failed_recipients: List[str] = field(default_factory=list)
    error: Optional[Exception] = None
    message: str = ""


@dataclass
class SendResult:
    success: bool
    message: str
    total_recipients: int = 0
    total_sent: int = 0
    total_failed: int = 0
    failed_recipients: List[str] = field(default_factory=list)
    batch_results: List[BatchResult] = field(default_factory=list)
    attachments: List[AttachmentInfo] = field(default_factory=list)
    error: Optional[Exception] = None

    def __bool__(self) -> bool:
        return self.success and self.total_failed == 0

    def add_failed(self, recipients: List[str]) -> None:
        self.failed_recipients.extend(recipients)
        self.total_failed += len(recipients)


class AttachmentValidator:
    @staticmethod
    def validate_file(file_path: str) -> AttachmentInfo:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Attachment not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > MAX_ATTACHMENT_SIZE:
            raise ValueError(
                f"Attachment '{os.path.basename(file_path)}' exceeds 10MB limit "
                f"({file_size / (1024 * 1024):.1f}MB)"
            )

        filename = os.path.basename(file_path)
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"

        return AttachmentInfo(filename=filename, size=file_size, content_type=content_type)

    @staticmethod
    def validate_bytes(data: bytes, filename: str) -> AttachmentInfo:
        file_size = len(data)
        if file_size > MAX_ATTACHMENT_SIZE:
            raise ValueError(
                f"Attachment '{filename}' exceeds 10MB limit "
                f"({file_size / (1024 * 1024):.1f}MB)"
            )

        content_type, _ = mimetypes.guess_type(filename)
        if content_type is None:
            content_type = "application/octet-stream"

        return AttachmentInfo(filename=filename, size=file_size, content_type=content_type)


class SMTPConnectionPool:
    def __init__(self, config: EmailConfig, max_retries: int = 3, retry_delay: float = 2.0):
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connection: Optional[smtplib.SMTP] = None

    def _connect(self) -> smtplib.SMTP:
        smtp = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port, timeout=30)
        smtp.ehlo()
        if self.config.use_tls:
            smtp.starttls()
            smtp.ehlo()
        smtp.login(self.config.smtp_username, self.config.smtp_password)
        return smtp

    def get_connection(self) -> smtplib.SMTP:
        if self._connection is None:
            self._connection = self._connect()
            logger.debug("Created new SMTP connection")
        else:
            try:
                self._connection.noop()
                logger.debug("Reusing existing SMTP connection")
            except (smtplib.SMTPException, OSError):
                logger.debug("Existing connection expired, creating new one")
                self._connection = self._connect()
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.quit()
            except Exception:
                pass
            self._connection = None
            logger.debug("SMTP connection closed")

    def __enter__(self) -> "SMTPConnectionPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class EmailSender:
    def __init__(self, config: EmailConfig, batch_size: int = DEFAULT_BATCH_SIZE, batch_delay: float = DEFAULT_BATCH_DELAY):
        self.config = config
        self.config.validate()
        self.batch_size = batch_size
        self.batch_delay = batch_delay

    def _add_attachment_from_file(self, msg: MIMEMultipart, file_path: str) -> AttachmentInfo:
        info = AttachmentValidator.validate_file(file_path)

        with open(file_path, "rb") as f:
            part = MIMEBase(*info.content_type.split("/", 1))
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=info.filename,
        )
        msg.attach(part)

        logger.debug(f"Attached file: {info.filename} ({info.size} bytes)")
        return info

    def _add_attachment_from_bytes(
        self, msg: MIMEMultipart, data: bytes, filename: str
    ) -> AttachmentInfo:
        info = AttachmentValidator.validate_bytes(data, filename)

        part = MIMEBase(*info.content_type.split("/", 1))
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=info.filename,
        )
        msg.attach(part)

        logger.debug(f"Attached bytes: {info.filename} ({info.size} bytes)")
        return info

    def _build_message(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        is_html: bool = False,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        inline_attachments: Optional[Dict[str, bytes]] = None,
    ) -> tuple:
        has_attachments = bool(attachments) or bool(inline_attachments)

        if has_attachments:
            msg = MIMEMultipart("mixed")
            msg["From"] = formataddr((self.config.sender_name, self.config.sender_email))
            msg["To"] = ", ".join(to) if isinstance(to, list) else to
            msg["Subject"] = subject
            msg["Date"] = formatdate(localtime=True)
            msg["Message-ID"] = make_msgid()

            if cc:
                msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
            if reply_to:
                msg["Reply-To"] = reply_to

            alt_part = MIMEMultipart("alternative")
            mime_type = "html" if is_html else "plain"
            alt_part.attach(MIMEText(body, mime_type, "utf-8"))
            msg.attach(alt_part)
        else:
            msg = MIMEMultipart("alternative")
            msg["From"] = formataddr((self.config.sender_name, self.config.sender_email))
            msg["To"] = ", ".join(to) if isinstance(to, list) else to
            msg["Subject"] = subject
            msg["Date"] = formatdate(localtime=True)
            msg["Message-ID"] = make_msgid()

            if cc:
                msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
            if reply_to:
                msg["Reply-To"] = reply_to

            mime_type = "html" if is_html else "plain"
            msg.attach(MIMEText(body, mime_type, "utf-8"))

        attachment_infos: List[AttachmentInfo] = []

        if attachments:
            for file_path in attachments:
                info = self._add_attachment_from_file(msg, file_path)
                attachment_infos.append(info)

        if inline_attachments:
            for filename, data in inline_attachments.items():
                info = self._add_attachment_from_bytes(msg, data, filename)
                attachment_infos.append(info)

        return msg, attachment_infos

    def _get_all_recipients(
        self,
        to: Union[str, List[str]],
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        recipients = []
        for addr_list in [to, cc, bcc]:
            if addr_list:
                if isinstance(addr_list, str):
                    recipients.append(addr_list)
                else:
                    recipients.extend(addr_list)
        return recipients

    def _split_into_batches(self, recipients: List[str]) -> List[List[str]]:
        return [recipients[i:i + self.batch_size] for i in range(0, len(recipients), self.batch_size)]

    def _send_single_batch(
        self,
        smtp: smtplib.SMTP,
        batch_recipients: List[str],
        msg: MIMEMultipart,
        batch_index: int,
    ) -> BatchResult:
        result = BatchResult(
            batch_index=batch_index,
            recipients=batch_recipients.copy(),
            success=False,
        )

        try:
            send_errors = smtp.sendmail(
                self.config.sender_email,
                batch_recipients,
                msg.as_string(),
            )

            if send_errors:
                result.success = len(send_errors) < len(batch_recipients)
                result.failed_recipients = list(send_errors.keys())
                result.message = f"Batch {batch_index}: Partial success, {len(result.failed_recipients)} failed"
                logger.warning(f"Batch {batch_index}: Failed recipients: {result.failed_recipients}")
            else:
                result.success = True
                result.message = f"Batch {batch_index}: All {len(batch_recipients)} recipients succeeded"
                logger.info(f"Batch {batch_index}: Sent successfully to {len(batch_recipients)} recipients")

        except smtplib.SMTPRecipientsRefused as e:
            result.failed_recipients = list(e.recipients.keys())
            result.error = e
            result.message = f"Batch {batch_index}: All recipients refused: {str(e)}"
            logger.error(result.message)
        except smtplib.SMTPException as e:
            result.error = e
            result.message = f"Batch {batch_index}: SMTP error: {str(e)}"
            logger.error(result.message)
        except Exception as e:
            result.error = e
            result.message = f"Batch {batch_index}: Unexpected error: {str(e)}"
            logger.exception(result.message)

        return result

    def send(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        is_html: bool = False,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        inline_attachments: Optional[Dict[str, bytes]] = None,
    ) -> SendResult:
        if not to:
            return SendResult(success=False, message="Recipient (to) is required")
        if not subject:
            return SendResult(success=False, message="Subject is required")
        if not body:
            return SendResult(success=False, message="Body is required")

        all_recipients = self._get_all_recipients(to, cc, bcc)
        total_recipients = len(all_recipients)

        result = SendResult(
            success=False,
            message="",
            total_recipients=total_recipients,
        )

        try:
            msg, attachment_infos = self._build_message(
                to, subject, body, is_html, cc, bcc, reply_to, attachments, inline_attachments
            )
            result.attachments = attachment_infos

            batches = self._split_into_batches(all_recipients)
            total_batches = len(batches)

            logger.info(f"Starting batch send: {total_recipients} recipients, {total_batches} batches of {self.batch_size}")

            with SMTPConnectionPool(self.config) as pool:
                smtp = pool.get_connection()

                for batch_idx, batch_recipients in enumerate(batches, 1):
                    logger.info(f"Processing batch {batch_idx}/{total_batches}: {len(batch_recipients)} recipients")

                    batch_result = self._send_single_batch(smtp, batch_recipients, msg, batch_idx)
                    result.batch_results.append(batch_result)

                    if batch_result.success:
                        result.total_sent += len(batch_recipients) - len(batch_result.failed_recipients)
                    if batch_result.failed_recipients:
                        result.add_failed(batch_result.failed_recipients)

                    if batch_idx < total_batches and self.batch_delay > 0:
                        logger.debug(f"Waiting {self.batch_delay}s before next batch...")
                        time.sleep(self.batch_delay)

            result.success = result.total_failed == 0
            if result.success:
                result.message = f"Successfully sent to all {result.total_sent} recipients in {total_batches} batches"
            else:
                result.message = f"Sent to {result.total_sent}/{total_recipients} recipients, {result.total_failed} failed"

            logger.info(result.message)
            return result

        except FileNotFoundError as e:
            error_msg = f"Attachment error: {str(e)}"
            logger.error(error_msg)
            result.error = e
            result.message = error_msg
            return result
        except ValueError as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            result.error = e
            result.message = error_msg
            return result
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {str(e)}"
            logger.error(error_msg)
            result.error = e
            result.message = error_msg
            return result
        except smtplib.SMTPConnectError as e:
            error_msg = f"SMTP connection error: {str(e)}"
            logger.error(error_msg)
            result.error = e
            result.message = error_msg
            return result
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.exception(error_msg)
            result.error = e
            result.message = error_msg
            return result

    def send_single(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        is_html: bool = False,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        inline_attachments: Optional[Dict[str, bytes]] = None,
    ) -> SendResult:
        return self.send(to, subject, body, is_html, cc, bcc, reply_to, attachments, inline_attachments)


def send_email(
    to: Union[str, List[str]],
    subject: str,
    body: str,
    is_html: bool = False,
    cc: Optional[Union[str, List[str]]] = None,
    bcc: Optional[Union[str, List[str]]] = None,
    reply_to: Optional[str] = None,
    config: Optional[EmailConfig] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_delay: float = DEFAULT_BATCH_DELAY,
    attachments: Optional[List[str]] = None,
    inline_attachments: Optional[Dict[str, bytes]] = None,
) -> SendResult:
    if config is None:
        config = EmailConfig.from_env()
    sender = EmailSender(config, batch_size=batch_size, batch_delay=batch_delay)
    return sender.send(to, subject, body, is_html, cc, bcc, reply_to, attachments, inline_attachments)
