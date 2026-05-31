import base64
import json
import requests


def encode_file_to_base64(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


API_BASE = 'http://localhost:5000/api'
HEALTH_URL = 'http://localhost:5000/health'


def test_health_check():
    print('Testing health check...')
    try:
        response = requests.get(HEALTH_URL)
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_send_email_basic():
    print('Testing basic email send...')
    payload = {
        'to': 'recipient@example.com',
        'subject': 'Test Email - Basic',
        'html_body': '''
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h1 style="color: #333;">Hello!</h1>
                <p>This is a test email sent via the SMTP API.</p>
                <p style="color: #666; font-size: 12px;">This is an automated message.</p>
            </body>
        </html>
        '''
    }

    try:
        response = requests.post(f'{API_BASE}/send-email', json=payload)
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_send_email_cc_bcc():
    print('Testing email with CC and BCC...')
    payload = {
        'to': 'recipient@example.com',
        'cc': ['cc1@example.com', 'cc2@example.com'],
        'bcc': ['bcc@example.com'],
        'subject': 'Test Email - With CC/BCC',
        'html_body': '''
        <div style="padding: 20px;">
            <h2>Email with CC and BCC</h2>
            <p>This email has CC and BCC recipients.</p>
        </div>
        '''
    }

    try:
        response = requests.post(f'{API_BASE}/send-email', json=payload)
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_send_email_with_template():
    print('Testing email with Jinja2 template...')
    payload = {
        'to': 'user@example.com',
        'subject': 'Welcome to Our Service',
        'template_name': 'welcome.html',
        'template_context': {
            'username': '张三',
            'verify_url': 'https://example.com/verify?token=abc123',
            'expire_hours': 24,
            'company_name': '科技公司',
            'year': 2024
        }
    }

    try:
        response = requests.post(f'{API_BASE}/send-email', json=payload)
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_send_email_with_inline_template():
    print('Testing email with inline template...')
    payload = {
        'to': 'user@example.com',
        'subject': 'Inline Template Test',
        'template_string': '''
        <div style="font-family: sans-serif;">
            <h1>Hello {{ name }}!</h1>
            <p>Your order #{{ order_id }} has been {{ status }}.</p>
            <ul>
            {% for item in items %}
                <li>{{ item.name }}: ${{ item.price }}</li>
            {% endfor %}
            </ul>
            <p>Total: ${{ total }}</p>
        </div>
        ''',
        'template_context': {
            'name': '李四',
            'order_id': 'ORD-001',
            'status': 'shipped',
            'items': [
                {'name': 'Product A', 'price': 99.99},
                {'name': 'Product B', 'price': 49.99}
            ],
            'total': 149.98
        }
    }

    try:
        response = requests.post(f'{API_BASE}/send-email', json=payload)
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_queue_email():
    print('Testing queue email...')
    payload = {
        'to': 'recipient@example.com',
        'subject': 'Queued Email Test',
        'html_body': '<h1>This is a queued email</h1><p>Processed by the email queue worker.</p>'
    }

    try:
        response = requests.post(f'{API_BASE}/queue-email', json=payload)
        print(f'Status: {response.status_code}')
        result = response.json()
        print(f'Response: {json.dumps(result, indent=2, ensure_ascii=False)}')
        return result.get('job_id')
    except Exception as e:
        print(f'Error: {e}')
    print()
    return None


def test_queue_stats():
    print('Testing queue stats...')
    try:
        response = requests.get(f'{API_BASE}/queue/stats')
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_job_status(job_id):
    if not job_id:
        print('Skipping job status test (no job id)')
        print()
        return
    
    print(f'Testing job status for {job_id}...')
    try:
        response = requests.get(f'{API_BASE}/queue/jobs/{job_id}')
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_list_templates():
    print('Testing list templates...')
    try:
        response = requests.get(f'{API_BASE}/templates')
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_render_template():
    print('Testing render template...')
    payload = {
        'template_name': 'welcome.html',
        'context': {
            'username': '测试用户',
            'verify_url': 'https://example.com/verify',
            'expire_hours': 24,
            'company_name': '科技公司',
            'year': 2024
        }
    }

    try:
        response = requests.post(f'{API_BASE}/templates/render', json=payload)
        print(f'Status: {response.status_code}')
        result = response.json()
        if result.get('success'):
            print('Template rendered successfully (first 200 chars):')
            print(result['rendered'][:200] + '...')
        else:
            print(f'Response: {json.dumps(result, indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_batch_send():
    print('Testing batch send...')
    payload = {
        'emails': [
            {
                'to': 'user1@example.com',
                'subject': 'Batch Email 1',
                'html_body': '<h1>Hello User 1</h1>'
            },
            {
                'to': 'user2@example.com',
                'subject': 'Batch Email 2',
                'template_string': '<p>Hi {{ name }}!</p>',
                'template_context': {'name': 'User 2'}
            },
            {
                'to': 'user3@example.com',
                'subject': 'Batch Email 3',
                'html_body': '<h1>Hello User 3</h1>'
            }
        ]
    }

    try:
        response = requests.post(f'{API_BASE}/batch-send', json=payload)
        print(f'Status: {response.status_code}')
        result = response.json()
        print(f'Response: {json.dumps(result, indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_batch_queue():
    print('Testing batch queue...')
    payload = {
        'emails': [
            {
                'to': 'batch1@example.com',
                'subject': 'Batch Queue Email 1',
                'html_body': '<h1>Batch Queue 1</h1>'
            },
            {
                'to': 'batch2@example.com',
                'subject': 'Batch Queue Email 2',
                'html_body': '<h1>Batch Queue 2</h1>'
            },
            {
                'to': 'batch3@example.com',
                'subject': 'Batch Queue Email 3',
                'html_body': '<h1>Batch Queue 3</h1>'
            }
        ]
    }

    try:
        response = requests.post(f'{API_BASE}/batch-queue', json=payload)
        print(f'Status: {response.status_code}')
        result = response.json()
        print(f'Response: {json.dumps(result, indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_send_email_chinese_filename():
    print('Testing email with Chinese filename attachment...')

    text_content = '这是中文文件名的测试文件内容！\n测试中文编码是否正常工作。'

    payload = {
        'to': 'recipient@example.com',
        'subject': 'Test Email - Chinese Filename',
        'html_body': '''
        <div style="font-family: sans-serif;">
            <h1>中文文件名测试</h1>
            <p>测试邮件中的中文附件文件名编码是否正常。</p>
        </div>
        ''',
        'attachments': [
            {
                'filename': '测试文档_中文文件名.txt',
                'content': base64.b64encode(text_content.encode('utf-8')).decode('utf-8'),
                'content_type': 'text/plain'
            },
            {
                'filename': '报告_2024.pdf',
                'content': base64.b64encode(b'%PDF-1.4 test content').decode('utf-8'),
                'content_type': 'application/pdf'
            }
        ]
    }

    try:
        response = requests.post(f'{API_BASE}/send-email', json=payload)
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


def test_send_email_validation():
    print('Testing validation errors...')
    payload = {
        'subject': 'Missing required fields'
    }

    try:
        response = requests.post(f'{API_BASE}/send-email', json=payload)
        print(f'Status: {response.status_code}')
        print(f'Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}')
    except Exception as e:
        print(f'Error: {e}')
    print()


if __name__ == '__main__':
    print('=' * 60)
    print('SMTP Email API Test Suite - Enhanced Features')
    print('=' * 60)
    print()

    test_health_check()
    test_send_email_validation()
    test_list_templates()
    test_render_template()
    test_send_email_basic()
    test_send_email_cc_bcc()
    test_send_email_with_template()
    test_send_email_with_inline_template()
    test_send_email_chinese_filename()
    
    job_id = test_queue_email()
    test_queue_stats()
    test_job_status(job_id)
    
    test_batch_send()
    test_batch_queue()

    print('=' * 60)
    print('Test suite completed!')
    print('=' * 60)
