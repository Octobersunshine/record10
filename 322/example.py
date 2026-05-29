import logging
import time
from email_sender import send_email, EmailSender, EmailConfig
from email_template import EmailTemplate, RenderedEmail
from email_queue import EmailQueue, TaskStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

html_body = """
<html>
  <body>
    <h1 style="color: #2c3e50;">Hello!</h1>
    <p>This is a <strong>HTML email</strong> sent via Python SMTP API.</p>
    <ul>
      <li>Supports HTML formatting</li>
      <li>Supports multiple recipients</li>
      <li>Returns send status</li>
    </ul>
    <p style="color: #7f8c8d; font-size: 12px;">Best regards,<br>Your App</p>
  </body>
</html>
"""

plain_body = """Hello!

This is a plain text email sent via Python SMTP API.

Supports:
- HTML formatting
- Multiple recipients
- Send status return

Best regards,
Your App
"""


def example_basic_send():
    print("=== Basic Send ===")
    result = send_email(
        to="recipient@example.com",
        subject="Test Email from Python API",
        body=plain_body,
        is_html=False,
    )
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Total recipients: {result.total_recipients}")
    print(f"Total sent: {result.total_sent}")
    print(f"Total failed: {result.total_failed}")
    if result.failed_recipients:
        print(f"Failed recipients: {result.failed_recipients}")
    print()


def example_html_send():
    print("=== HTML Send ===")
    result = send_email(
        to=["user1@example.com", "user2@example.com"],
        subject="HTML Test Email",
        body=html_body,
        is_html=True,
        cc="cc@example.com",
        reply_to="support@example.com",
    )
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Total batches: {len(result.batch_results)}")
    print()


def example_with_custom_config():
    print("=== Custom Config ===")
    config = EmailConfig(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        smtp_username="your-email@gmail.com",
        smtp_password="your-app-password",
        use_tls=True,
        sender_name="My App",
        sender_email="your-email@gmail.com",
    )
    sender = EmailSender(config, batch_size=10, batch_delay=2.0)
    result = sender.send(
        to="recipient@example.com",
        subject="Test with Custom Config",
        body="This uses a custom config object.",
    )
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print()


def example_result_check():
    print("=== Result Check ===")
    result = send_email(
        to="test@example.com",
        subject="Testing boolean check",
        body="Checking if result works as boolean.",
    )
    if result:
        print("Email was sent successfully!")
    else:
        print(f"Send failed: {result.message}")
        if result.error:
            print(f"Error details: {result.error}")
    print()


def example_batch_sending():
    print("=== Batch Sending Example (150 recipients) ===")
    many_recipients = [f"user{i}@example.com" for i in range(1, 151)]

    result = send_email(
        to=many_recipients,
        subject="Bulk Email - Batch Test",
        body="<h1>Hello!</h1><p>This is a batch test email sent to many recipients.</p>",
        is_html=True,
        batch_size=20,
        batch_delay=1.5,
    )

    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Total recipients: {result.total_recipients}")
    print(f"Total sent: {result.total_sent}")
    print(f"Total failed: {result.total_failed}")
    print(f"Number of batches: {len(result.batch_results)}")

    if result.failed_recipients:
        print(f"\nFailed recipients ({len(result.failed_recipients)}):")
        for email in result.failed_recipients:
            print(f"  - {email}")

    print("\nBatch details:")
    for batch in result.batch_results:
        status = "OK" if batch.success and not batch.failed_recipients else "WARN" if batch.failed_recipients else "FAIL"
        print(f"  [{status}] Batch {batch.batch_index}: {len(batch.recipients)} recipients, {len(batch.failed_recipients)} failed")
        if batch.failed_recipients:
            print(f"      Failed: {', '.join(batch.failed_recipients)}")
    print()


def example_template_rendering():
    print("=== Jinja2 Template Rendering ===")
    template_engine = EmailTemplate()

    subject_tpl = "Welcome to {{ app_name }}, {{ username }}!"
    body_tpl = """
    <html>
      <body>
        <h2>Hello, {{ username }}!</h2>
        <p>Thank you for joining <strong>{{ app_name }}</strong>.</p>
        <p>Your account was created on {{ signup_date }}.</p>
        {% if is_premium %}
        <p style="color: gold;">You are a Premium member! Enjoy exclusive features.</p>
        {% endif %}
        <p>Best regards,<br>The {{ app_name }} Team</p>
      </body>
    </html>
    """

    context = {
        "username": "Alice",
        "app_name": "MyApp",
        "signup_date": "2026-05-29",
        "is_premium": True,
    }

    rendered = template_engine.render_email(
        subject_template=subject_tpl,
        body_template=body_tpl,
        context=context,
        is_html=True,
    )

    print(f"Rendered Subject: {rendered.subject}")
    print(f"Rendered Body (first 200 chars): {rendered.body[:200]}...")
    print(f"Is HTML: {rendered.is_html}")
    print()

    users = [
        {"username": "Alice", "app_name": "MyApp", "signup_date": "2026-05-29", "is_premium": True},
        {"username": "Bob", "app_name": "MyApp", "signup_date": "2026-05-28", "is_premium": False},
    ]

    print("Batch template rendering for multiple users:")
    for user_ctx in users:
        rendered = template_engine.render_email(
            subject_template=subject_tpl,
            body_template=body_tpl,
            context=user_ctx,
            is_html=True,
        )
        print(f"  - {rendered.subject}")
    print()


def example_attachment_send():
    print("=== Attachment Send ===")

    result = send_email(
        to="recipient@example.com",
        subject="Email with Attachment",
        body="<h1>Hello!</h1><p>Please find the attached file.</p>",
        is_html=True,
        attachments=["./report.pdf", "./data.csv"],
    )

    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.attachments:
        print(f"Attachments ({len(result.attachments)}):")
        for att in result.attachments:
            print(f"  - {att.filename} ({att.size} bytes, {att.content_type})")
    print()


def example_inline_attachment():
    print("=== Inline Attachment (from bytes) ===")

    csv_data = b"id,name,email\n1,Alice,alice@example.com\n2,Bob,bob@example.com"
    json_data = b'{"status": "ok", "count": 2}'

    result = send_email(
        to="recipient@example.com",
        subject="Email with Inline Attachments",
        body="<h1>Report</h1><p>See attached files.</p>",
        is_html=True,
        inline_attachments={
            "report.csv": csv_data,
            "data.json": json_data,
        },
    )

    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.attachments:
        for att in result.attachments:
            print(f"  - {att.filename} ({att.size} bytes)")
    print()


def example_attachment_size_limit():
    print("=== Attachment Size Limit Test ===")
    config = EmailConfig(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        smtp_username="test@gmail.com",
        smtp_password="password",
        sender_email="test@gmail.com",
    )
    sender = EmailSender(config)

    from email_sender import AttachmentValidator
    try:
        big_data = b"x" * (11 * 1024 * 1024)
        AttachmentValidator.validate_bytes(big_data, "huge_file.bin")
    except ValueError as e:
        print(f"Expected error caught: {e}")

    small_data = b"x" * 1024
    info = AttachmentValidator.validate_bytes(small_data, "small_file.txt")
    print(f"Small file validated: {info.filename} ({info.size} bytes)")
    print()


def example_async_queue():
    print("=== Async Email Queue ===")
    config = EmailConfig.from_env()

    with EmailQueue(config, batch_size=20, batch_delay=0.5, max_workers=2) as queue:
        task_id_1 = queue.submit(
            to=["user1@example.com", "user2@example.com"],
            subject="Async Email 1",
            body="<h1>First async email</h1>",
            is_html=True,
        )
        task_id_2 = queue.submit(
            to="user3@example.com",
            subject="Async Email 2",
            body="Second async email (plain text)",
            is_html=False,
        )
        task_id_3 = queue.submit(
            to=["user4@example.com", "user5@example.com"],
            subject="Async Email with Attachment",
            body="<p>See attached report.</p>",
            is_html=True,
            attachments=["./report.pdf"],
        )

        print(f"Submitted tasks: {task_id_1[:8]}..., {task_id_2[:8]}..., {task_id_3[:8]}...")

        print("\nPolling task statuses...")
        for i in range(10):
            time.sleep(1)
            stats = queue.get_stats()
            print(f"  [{i+1}s] Pending: {stats.pending}, Running: {stats.running}, "
                  f"Completed: {stats.completed}, Failed: {stats.failed}")
            if stats.pending == 0 and stats.running == 0:
                break

        print("\nFinal task results:")
        for task_id in [task_id_1, task_id_2, task_id_3]:
            task = queue.get_status(task_id)
            if task:
                print(f"  Task {task_id[:8]}...: {task.status.value} "
                      f"(duration: {task.duration:.2f}s)" if task.duration else f"  Task {task_id[:8]}...: {task.status.value}")
                if task.result:
                    print(f"    -> {task.result.message}")
                    if task.result.failed_recipients:
                        print(f"    Failed: {task.result.failed_recipients}")
    print()


def example_template_with_queue():
    print("=== Template + Queue Combined ===")
    config = EmailConfig.from_env()
    template_engine = EmailTemplate()

    subject_tpl = "Monthly Report - {{ month }} {{ year }}"
    body_tpl = """
    <html>
      <body>
        <h2>Monthly Report</h2>
        <p>Dear {{ username }},</p>
        <p>Your report for {{ month }} {{ year }} is ready.</p>
        <p>Summary: {{ total_sales }} sales, ${{ revenue }} revenue.</p>
      </body>
    </html>
    """

    recipients_data = [
        {"username": "Alice", "email": "alice@example.com", "total_sales": 150, "revenue": "12,500"},
        {"username": "Bob", "email": "bob@example.com", "total_sales": 98, "revenue": "8,300"},
        {"username": "Charlie", "email": "charlie@example.com", "total_sales": 210, "revenue": "18,900"},
    ]

    common_ctx = {"month": "May", "year": "2026"}

    with EmailQueue(config, max_workers=2) as queue:
        task_ids = []
        for user in recipients_data:
            ctx = {**common_ctx, **user}
            rendered = template_engine.render_email(
                subject_template=subject_tpl,
                body_template=body_tpl,
                context=ctx,
            )
            tid = queue.submit(
                to=user["email"],
                subject=rendered.subject,
                body=rendered.body,
                is_html=rendered.is_html,
            )
            task_ids.append(tid)
            print(f"  Queued email for {user['username']}: {rendered.subject}")

        time.sleep(5)

        print("\nResults:")
        for tid in task_ids:
            task = queue.get_status(tid)
            if task and task.result:
                print(f"  Task {tid[:8]}...: {task.status.value} - {task.result.message}")
    print()


if __name__ == "__main__":
    print("Email Sender API Examples")
    print("=" * 60)

    example_basic_send()
    example_html_send()
    example_with_custom_config()
    example_result_check()
    example_batch_sending()
    example_template_rendering()
    example_attachment_send()
    example_inline_attachment()
    example_attachment_size_limit()
    example_async_queue()
    example_template_with_queue()
