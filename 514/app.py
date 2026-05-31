from flask import Flask, request, jsonify
from mailer import send_email, queue_email, email_queue
from templates import render_email_template, get_available_templates
from config import Config

app = Flask(__name__)


def validate_email_request(data):
    errors = []

    if not data:
        errors.append('Request body is required')
        return errors

    if 'to' not in data or not data['to']:
        errors.append('Recipient email(s) are required')
    else:
        to = data['to']
        if isinstance(to, list) and len(to) == 0:
            errors.append('At least one recipient is required')

    if 'subject' not in data or not data['subject'].strip():
        errors.append('Email subject is required')

    has_html_body = 'html_body' in data and data['html_body'].strip()
    has_template = 'template_name' in data and data['template_name']
    has_template_string = 'template_string' in data and data['template_string']

    if not has_html_body and not has_template and not has_template_string:
        errors.append('Either html_body, template_name, or template_string is required')

    if 'attachments' in data and data['attachments']:
        attachments = data['attachments']
        if not isinstance(attachments, list):
            errors.append('Attachments must be a list')
        else:
            for idx, att in enumerate(attachments):
                if 'filename' not in att or not att['filename']:
                    errors.append(f'Attachment {idx + 1} is missing filename')
                if 'content' not in att or not att['content']:
                    errors.append(f'Attachment {idx + 1} is missing base64 content')

    return errors


def render_email_body(data):
    template_name = data.get('template_name')
    template_string = data.get('template_string')
    template_context = data.get('template_context', {})

    if template_name:
        return render_email_template(template_name=template_name, context=template_context)
    elif template_string:
        return render_email_template(template_string=template_string, context=template_context)
    else:
        return data.get('html_body', '')


@app.route('/health', methods=['GET'])
def health_check():
    queue_stats = email_queue.get_queue_stats()
    return jsonify({
        'status': 'healthy',
        'service': 'SMTP Email API',
        'queue': queue_stats
    }), 200


@app.route('/api/send-email', methods=['POST'])
def api_send_email():
    try:
        data = request.get_json()

        errors = validate_email_request(data)
        if errors:
            return jsonify({
                'success': False,
                'message': 'Validation failed',
                'errors': errors
            }), 400

        to_emails = data['to']
        subject = data['subject']
        cc_emails = data.get('cc', [])
        bcc_emails = data.get('bcc', [])
        attachments = data.get('attachments', [])

        try:
            html_body = render_email_body(data)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Template rendering failed: {str(e)}',
                'error_code': 'TEMPLATE_ERROR'
            }), 400

        result = send_email(to_emails, subject, html_body, cc_emails, bcc_emails, attachments)

        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/queue-email', methods=['POST'])
def api_queue_email():
    try:
        data = request.get_json()

        errors = validate_email_request(data)
        if errors:
            return jsonify({
                'success': False,
                'message': 'Validation failed',
                'errors': errors
            }), 400

        to_emails = data['to']
        subject = data['subject']
        cc_emails = data.get('cc', [])
        bcc_emails = data.get('bcc', [])
        attachments = data.get('attachments', [])
        template_name = data.get('template_name')
        template_context = data.get('template_context', {})

        try:
            html_body = render_email_body(data)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Template rendering failed: {str(e)}',
                'error_code': 'TEMPLATE_ERROR'
            }), 400

        job_id = queue_email(
            to_emails=to_emails,
            subject=subject,
            html_body=html_body,
            cc_emails=cc_emails,
            bcc_emails=bcc_emails,
            attachments=attachments,
            template_name=template_name,
            template_context=template_context
        )

        return jsonify({
            'success': True,
            'message': 'Email queued successfully',
            'job_id': job_id
        }), 202

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/queue/stats', methods=['GET'])
def api_queue_stats():
    stats = email_queue.get_queue_stats()
    return jsonify({
        'success': True,
        'stats': stats
    }), 200


@app.route('/api/queue/jobs', methods=['GET'])
def api_queue_jobs():
    jobs = email_queue.get_all_jobs()
    return jsonify({
        'success': True,
        'jobs': jobs
    }), 200


@app.route('/api/queue/jobs/<job_id>', methods=['GET'])
def api_job_status(job_id):
    job = email_queue.get_job_status(job_id)
    if job:
        return jsonify({
            'success': True,
            'job': job
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': 'Job not found'
        }), 404


@app.route('/api/queue/retry-failed', methods=['POST'])
def api_retry_failed():
    retried = email_queue.retry_failed_jobs()
    return jsonify({
        'success': True,
        'message': f'Retried {len(retried)} failed jobs',
        'retried_jobs': retried
    }), 200


@app.route('/api/templates', methods=['GET'])
def api_list_templates():
    templates = get_available_templates()
    return jsonify({
        'success': True,
        'templates': templates
    }), 200


@app.route('/api/templates/render', methods=['POST'])
def api_render_template():
    try:
        data = request.get_json()
        template_name = data.get('template_name')
        template_string = data.get('template_string')
        context = data.get('context', {})

        if not template_name and not template_string:
            return jsonify({
                'success': False,
                'message': 'Either template_name or template_string is required'
            }), 400

        rendered = render_email_template(
            template_name=template_name,
            template_string=template_string,
            context=context
        )

        return jsonify({
            'success': True,
            'rendered': rendered
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Template rendering failed: {str(e)}',
            'error_code': 'TEMPLATE_ERROR'
        }), 400


@app.route('/api/batch-send', methods=['POST'])
def api_batch_send():
    try:
        data = request.get_json()

        if not data or 'emails' not in data:
            return jsonify({
                'success': False,
                'message': 'Batch emails list is required'
            }), 400

        emails = data['emails']
        if not isinstance(emails, list):
            return jsonify({
                'success': False,
                'message': 'emails must be a list'
            }), 400

        results = []
        for idx, email_data in enumerate(emails):
            errors = validate_email_request(email_data)
            if errors:
                results.append({
                    'index': idx,
                    'success': False,
                    'message': 'Validation failed',
                    'errors': errors
                })
                continue

            try:
                html_body = render_email_body(email_data)
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'message': f'Template rendering failed: {str(e)}',
                    'error_code': 'TEMPLATE_ERROR'
                })
                continue

            result = send_email(
                to_emails=email_data['to'],
                subject=email_data['subject'],
                html_body=html_body,
                cc_emails=email_data.get('cc', []),
                bcc_emails=email_data.get('bcc', []),
                attachments=email_data.get('attachments', [])
            )
            result['index'] = idx
            results.append(result)

        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count

        return jsonify({
            'success': True,
            'message': f'Batch completed: {success_count} succeeded, {failed_count} failed',
            'total': len(results),
            'success_count': success_count,
            'failed_count': failed_count,
            'results': results
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/batch-queue', methods=['POST'])
def api_batch_queue():
    try:
        data = request.get_json()

        if not data or 'emails' not in data:
            return jsonify({
                'success': False,
                'message': 'Batch emails list is required'
            }), 400

        emails = data['emails']
        if not isinstance(emails, list):
            return jsonify({
                'success': False,
                'message': 'emails must be a list'
            }), 400

        job_ids = []
        errors = []

        for idx, email_data in enumerate(emails):
            validation_errors = validate_email_request(email_data)
            if validation_errors:
                errors.append({
                    'index': idx,
                    'errors': validation_errors
                })
                continue

            try:
                html_body = render_email_body(email_data)
            except Exception as e:
                errors.append({
                    'index': idx,
                    'error': f'Template rendering failed: {str(e)}'
                })
                continue

            job_id = queue_email(
                to_emails=email_data['to'],
                subject=email_data['subject'],
                html_body=html_body,
                cc_emails=email_data.get('cc', []),
                bcc_emails=email_data.get('bcc', []),
                attachments=email_data.get('attachments', []),
                template_name=email_data.get('template_name'),
                template_context=email_data.get('template_context', {})
            )
            job_ids.append({
                'index': idx,
                'job_id': job_id
            })

        return jsonify({
            'success': True,
            'message': f'Queued {len(job_ids)} emails successfully',
            'queued': len(job_ids),
            'failed': len(errors),
            'job_ids': job_ids,
            'errors': errors
        }), 202

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


if __name__ == '__main__':
    email_queue.start()
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=Config.API_DEBUG
    )
