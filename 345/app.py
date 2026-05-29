import os
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from middleware import FlaskRequestLogger
from storage import DatabaseLogStorage
from alerting import create_default_alert_manager


def create_app():
    app = Flask(__name__)

    db_storage = DatabaseLogStorage(
        db_path="logs/api_requests.db",
        table_name="api_request_logs",
    )

    alert_manager = create_default_alert_manager(
        error_threshold=10,
        window_seconds=300,
        cooldown_seconds=300,
        alert_log_file="logs/alerts.log",
    )

    request_logger = FlaskRequestLogger(
        app,
        storage=db_storage,
        use_file=True,
        file_config={"log_dir": "logs", "filename": "api_requests.log"},
        sensitive_fields=[
            "password", "pwd", "secret", "token", "authorization",
            "credit_card", "cvv", "ssn", "id_card", "phone", "email",
        ],
        exclude_paths=["/health"],
        exclude_methods=["OPTIONS"],
        anonymize_ip=False,
        log_response_body=True,
        alert_manager=alert_manager,
    )

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy"}), 200

    @app.route("/api/users", methods=["GET"])
    def get_users():
        return jsonify({
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }), 200

    @app.route("/api/users/<int:user_id>", methods=["GET"])
    def get_user(user_id):
        return jsonify({"id": user_id, "name": "Alice"}), 200

    @app.route("/api/users", methods=["POST"])
    def create_user():
        data = request.get_json()
        return jsonify({
            "id": 3,
            "name": data.get("name"),
            "email": data.get("email"),
        }), 201

    @app.route("/api/login", methods=["POST"])
    def login():
        data = request.get_json()
        if data and data.get("username") == "admin" and data.get("password"):
            return jsonify({"token": "fake-jwt-token"}), 200
        return jsonify({"error": "Invalid credentials"}), 401

    @app.route("/api/payment", methods=["POST"])
    def payment():
        data = request.get_json()
        return jsonify({"status": "success", "transaction_id": "txn_123"}), 200

    @app.route("/api/error", methods=["GET"])
    def error_endpoint():
        raise RuntimeError("Something went wrong!")

    @app.route("/api/logs/query", methods=["GET"])
    def query_logs():
        ip = request.args.get("ip")
        path = request.args.get("path")
        method = request.args.get("method")
        status_code = request.args.get("status_code", type=int)
        status_code_min = request.args.get("status_code_min", type=int)
        status_code_max = request.args.get("status_code_max", type=int)
        start_time = request.args.get("start_time")
        end_time = request.args.get("end_time")
        limit = request.args.get("limit", default=100, type=int)
        offset = request.args.get("offset", default=0, type=int)

        start_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
            except ValueError:
                return jsonify({"error": "Invalid start_time format. Use ISO 8601."}), 400

        end_dt = None
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time)
            except ValueError:
                return jsonify({"error": "Invalid end_time format. Use ISO 8601."}), 400

        results = db_storage.query(
            ip=ip,
            path=path,
            method=method,
            status_code=status_code,
            status_code_min=status_code_min,
            status_code_max=status_code_max,
            start_time=start_dt,
            end_time=end_dt,
            limit=min(limit, 1000),
            offset=offset,
        )

        total = db_storage.count(
            ip=ip,
            path=path,
            method=method,
            status_code=status_code,
            status_code_min=status_code_min,
            status_code_max=status_code_max,
            start_time=start_dt,
            end_time=end_dt,
        )

        return jsonify({
            "total": total,
            "limit": limit,
            "offset": offset,
            "results": results,
        })

    @app.route("/api/logs/export", methods=["GET"])
    def export_logs():
        ip = request.args.get("ip")
        path = request.args.get("path")
        method = request.args.get("method")
        status_code = request.args.get("status_code", type=int)
        status_code_min = request.args.get("status_code_min", type=int)
        status_code_max = request.args.get("status_code_max", type=int)
        start_time = request.args.get("start_time")
        end_time = request.args.get("end_time")

        start_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
            except ValueError:
                return jsonify({"error": "Invalid start_time format."}), 400

        end_dt = None
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time)
            except ValueError:
                return jsonify({"error": "Invalid end_time format."}), 400

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"api_logs_{timestamp_str}.csv"
        csv_path = os.path.join("logs", "exports", csv_filename)

        db_storage.export_csv(
            output_path=csv_path,
            ip=ip,
            path=path,
            method=method,
            status_code=status_code,
            status_code_min=status_code_min,
            status_code_max=status_code_max,
            start_time=start_dt,
            end_time=end_dt,
        )

        return send_file(csv_path, as_attachment=True, download_name=csv_filename, mimetype="text/csv")

    @app.route("/api/logs/stats", methods=["GET"])
    def log_stats():
        start_time = request.args.get("start_time")
        end_time = request.args.get("end_time")

        start_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
            except ValueError:
                return jsonify({"error": "Invalid start_time format."}), 400

        end_dt = None
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time)
            except ValueError:
                return jsonify({"error": "Invalid end_time format."}), 400

        distribution = db_storage.get_status_distribution(
            start_time=start_dt,
            end_time=end_dt,
        )

        total = db_storage.count(
            start_time=start_dt,
            end_time=end_dt,
        )

        return jsonify({
            "total_requests": total,
            "status_distribution": distribution,
        })

    @app.route("/api/alerts", methods=["GET"])
    def get_alerts():
        limit = request.args.get("limit", default=50, type=int)
        return jsonify({
            "alerts": alert_manager.get_alert_history(limit=min(limit, 200)),
        })

    @app.route("/api/alerts/rules", methods=["GET"])
    def get_alert_rules():
        return jsonify({"rules": alert_manager.get_rule_status()})

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    print("Starting server on http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /health - Health check (excluded from logging)")
    print("  GET  /api/users - Get user list")
    print("  GET  /api/users/<id> - Get user by ID")
    print("  POST /api/users - Create user")
    print("  POST /api/login - Login (password will be masked)")
    print("  POST /api/payment - Payment (credit card will be masked)")
    print("  GET  /api/error - Test error handling")
    print("")
    print("  Log Query & Export:")
    print("  GET  /api/logs/query - Query logs (params: ip, path, method, status_code, start_time, end_time, limit, offset)")
    print("  GET  /api/logs/export - Export logs as CSV (same query params)")
    print("  GET  /api/logs/stats - Get log statistics")
    print("")
    print("  Alert Management:")
    print("  GET  /api/alerts - Get alert history")
    print("  GET  /api/alerts/rules - Get alert rules")
    app.run(debug=True, port=5000)
