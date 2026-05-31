import smtplib
import base64
import logging
import time
import os
import urllib.parse
import threading
import queue
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
from email.utils import formataddr, parseaddr
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailJob:
    def __init__(self, to_emails, subject, html_body, cc_emails=None, bcc_emails=None, 
                 attachments=None, template_name=None, template_context=None, job_id=None):
        self.job_id = job_id or str(uuid.uuid4())
        self.to_emails = to_emails if isinstance(to_emails, list) else [to_emails]
        self.cc_emails = cc_emails or []
        self.bcc_emails = bcc_emails or []
        self.subject = subject
        self.html_body = html_body
        self.attachments = attachments or []
        self.template_name = template_name
        self.template_context = template_context
        self.status = 'pending'
        self.retry_count = 0
        self.created_at = datetime.now()
        self.sent_at = None
        self.error = None

    def get_all_recipients(self):
        return list(set(self.to_emails + self.cc_emails + self.bcc_emails))

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'to': self.to_emails,
            'cc': self.cc_emails,
            'bcc': self.bcc_emails,
            'subject': self.subject,
            'status': self.status,
            'retry_count': self.retry_count,
            'created_at': self.created_at.isoformat(),
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'error': self.error
        }


class EmailSender:
    def __init__(self, config=None):
        self.config = config or Config()
        self._server = None

    def _format_email_address(self, name, email):
        name, email_addr = parseaddr(email)
        return formataddr((
            str(Header(name, 'utf-8')),
            email_addr
        ))

    def _decode_attachment(self, attachment):
        filename = attachment.get('filename')
        content_base64 = attachment.get('content')
        content_type = attachment.get('content_type', 'application/octet-stream')

        if not filename or not content_base64:
            return None

        try:
            file_content = base64.b64decode(content_base64)
        except Exception as e:
            logger.error(f"Failed to decode attachment {filename}: {e}")
            return None

        main_type, sub_type = content_type.split('/', 1) if '/' in content_type else ('application', 'octet-stream')

        part = MIMEBase(main_type, sub_type)
        part.set_payload(file_content)
        encoders.encode_base64(part)

        try:
            filename.encode('ascii')
            part.add_header(
                'Content-Disposition',
                'attachment',
                filename=filename
            )
        except UnicodeEncodeError:
            rfc2231_filename = urllib.parse.quote(filename, safe='')
            part.add_header(
                'Content-Disposition',
                'attachment',
                filename=f"UTF-8''{rfc2231_filename}"
            )

        return part

    def _build_message(self, job):
        msg = MIMEMultipart('alternative')
        msg['From'] = self._format_email_address(
            self.config.SMTP_FROM_NAME,
            self.config.SMTP_FROM_EMAIL
        )

        msg['To'] = ', '.join(job.to_emails)
        
        if job.cc_emails:
            msg['Cc'] = ', '.join(job.cc_emails)
        
        msg['Subject'] = Header(job.subject, 'utf-8')

        html_part = MIMEText(job.html_body, 'html', 'utf-8')
        msg.attach(html_part)

        total_attachment_size = 0
        if job.attachments:
            for attachment in job.attachments:
                part = self._decode_attachment(attachment)
                if part:
                    if 'content' in attachment:
                        try:
                            content_size = len(base64.b64decode(attachment['content']))
                            total_attachment_size += content_size
                        except:
                            pass
                    msg.attach(part)

        if total_attachment_size > 0:
            logger.info(f"Total attachment size: {total_attachment_size / (1024 * 1024):.2f} MB")
            if total_attachment_size > self.config.SMTP_LARGE_ATTACHMENT_THRESHOLD:
                logger.warning(f"Large attachment detected: {total_attachment_size / (1024 * 1024):.2f} MB")

        return msg

    def _connect_smtp(self):
        logger.info(f"Connecting to SMTP server {self.config.SMTP_SERVER}:{self.config.SMTP_PORT}")
        
        timeout = self.config.SMTP_TIMEOUT
        
        if self.config.SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(
                self.config.SMTP_SERVER, 
                self.config.SMTP_PORT, 
                timeout=timeout
            )
        else:
            server = smtplib.SMTP(
                self.config.SMTP_SERVER, 
                self.config.SMTP_PORT, 
                timeout=timeout
            )
            if self.config.SMTP_USE_TLS:
                server.starttls()

        server.login(self.config.SMTP_USERNAME, self.config.SMTP_PASSWORD)
        logger.info("SMTP connected and authenticated successfully")
        return server

    def _disconnect_smtp(self, server):
        if server:
            try:
                server.quit()
                logger.info("SMTP connection closed")
            except Exception as e:
                logger.warning(f"Error closing SMTP connection: {e}")

    def _send_with_retry(self, job):
        max_retries = self.config.SMTP_MAX_RETRIES
        retry_delay = self.config.SMTP_RETRY_DELAY
        
        server = None
        all_recipients = job.get_all_recipients()
        
        for attempt in range(max_retries):
            job.retry_count = attempt + 1
            try:
                if server is None:
                    server = self._connect_smtp()
                
                msg = self._build_message(job)
                server.sendmail(self.config.SMTP_FROM_EMAIL, all_recipients, msg.as_string())
                logger.info(f"Email sent successfully on attempt {attempt + 1}")
                
                job.status = 'completed'
                job.sent_at = datetime.now()
                return True, None
                
            except smtplib.SMTPAuthenticationError as e:
                job.error = f'SMTP Authentication failed: {str(e)}'
                job.status = 'failed'
                logger.error(f"SMTP Authentication failed: {e}")
                self._disconnect_smtp(server)
                return False, {
                    'success': False,
                    'message': job.error,
                    'error_code': 'AUTH_ERROR'
                }
                
            except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, 
                    smtplib.SMTPException, TimeoutError, ConnectionError) as e:
                job.error = f'SMTP error: {str(e)}'
                logger.warning(f"SMTP error on attempt {attempt + 1}/{max_retries}: {e}")
                
                self._disconnect_smtp(server)
                server = None
                
                if attempt < max_retries - 1:
                    job.status = 'retrying'
                    sleep_time = retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    job.status = 'failed'
                    logger.error(f"Failed after {max_retries} attempts")
                    return False, {
                        'success': False,
                        'message': f'Failed after {max_retries} retries: {str(e)}',
                        'error_code': 'RETRY_EXHAUSTED'
                    }
                    
            except Exception as e:
                job.error = f'Unexpected error: {str(e)}'
                job.status = 'failed'
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                self._disconnect_smtp(server)
                return False, {
                    'success': False,
                    'message': job.error,
                    'error_code': 'UNKNOWN_ERROR'
                }
        
        self._disconnect_smtp(server)
        return False, {
            'success': False,
            'message': f'Failed after {max_retries} retries',
            'error_code': 'RETRY_EXHAUSTED'
        }

    def send(self, to_emails, subject, html_body, cc_emails=None, bcc_emails=None, attachments=None):
        try:
            job = EmailJob(
                to_emails=to_emails,
                subject=subject,
                html_body=html_body,
                cc_emails=cc_emails,
                bcc_emails=bcc_emails,
                attachments=attachments
            )

            success, result = self._send_with_retry(job)
            
            if success:
                return {
                    'success': True,
                    'message': 'Email sent successfully',
                    'job_id': job.job_id,
                    'recipients': job.get_all_recipients(),
                    'subject': subject
                }
            else:
                return result

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {
                'success': False,
                'message': f'Failed to send email: {str(e)}',
                'error_code': 'UNKNOWN_ERROR'
            }

    def send_job(self, job):
        success, result = self._send_with_retry(job)
        return success, result


class EmailQueue:
    def __init__(self, config=None, max_workers=2):
        self.config = config or Config()
        self.sender = EmailSender(config)
        self.queue = queue.Queue()
        self.jobs = {}
        self.failed_jobs = []
        self.workers = []
        self.max_workers = max_workers
        self._running = False
        self._lock = threading.Lock()

    def enqueue(self, to_emails, subject, html_body, cc_emails=None, bcc_emails=None, 
                attachments=None, template_name=None, template_context=None):
        job = EmailJob(
            to_emails=to_emails,
            subject=subject,
            html_body=html_body,
            cc_emails=cc_emails,
            bcc_emails=bcc_emails,
            attachments=attachments,
            template_name=template_name,
            template_context=template_context
        )
        
        with self._lock:
            self.jobs[job.job_id] = job
        
        self.queue.put(job)
        logger.info(f"Email job {job.job_id} enqueued")
        return job.job_id

    def _worker(self):
        while self._running:
            try:
                job = self.queue.get(timeout=1)
                if job is None:
                    break
                
                logger.info(f"Processing email job {job.job_id}")
                success, result = self.sender.send_job(job)
                
                if not success and job.retry_count < self.config.SMTP_MAX_RETRIES:
                    self.failed_jobs.append(job)
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def start(self):
        if self._running:
            return
        
        self._running = True
        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)
        logger.info(f"Email queue started with {self.max_workers} workers")

    def stop(self):
        self._running = False
        for t in self.workers:
            t.join()
        self.workers = []
        logger.info("Email queue stopped")

    def get_job_status(self, job_id):
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                return job.to_dict()
        return None

    def get_all_jobs(self):
        with self._lock:
            return [job.to_dict() for job in self.jobs.values()]

    def retry_failed_jobs(self):
        retried = []
        for job in self.failed_jobs[:]:
            job.retry_count = 0
            job.status = 'pending'
            job.error = None
            self.queue.put(job)
            self.failed_jobs.remove(job)
            retried.append(job.job_id)
        logger.info(f"Retried {len(retried)} failed jobs")
        return retried

    def get_queue_stats(self):
        return {
            'pending': self.queue.qsize(),
            'total_jobs': len(self.jobs),
            'failed_jobs': len(self.failed_jobs),
            'workers': len(self.workers),
            'running': self._running
        }


email_sender = EmailSender()
email_queue = EmailQueue()


def send_email(to_emails, subject, html_body, cc_emails=None, bcc_emails=None, attachments=None):
    return email_sender.send(to_emails, subject, html_body, cc_emails, bcc_emails, attachments)


def queue_email(to_emails, subject, html_body, cc_emails=None, bcc_emails=None, 
                attachments=None, template_name=None, template_context=None):
    if not email_queue._running:
        email_queue.start()
    return email_queue.enqueue(to_emails, subject, html_body, cc_emails, bcc_emails, 
                               attachments, template_name, template_context)
