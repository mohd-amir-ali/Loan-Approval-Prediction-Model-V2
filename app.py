#!/usr/bin/env python3
"""
CRED-TREE // Explainable CART Decision Tree Loan Approval Web Application
=========================================================================
Data Science Internship Project — Mohd Amir Ali (160920733019)
Supports both Flask (when installed) and a zero-dependency Python HTTP server
fallback on Port 5065, while maintaining 100% backward compatibility with the
original POST /predict (box1..box9) form interface.
"""

import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from loan_cart_engine import (
    SAMPLE_PRESETS,
    evaluate_cart_decision_tree,
    get_dataset_eda_summary,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", 5065))


def parse_box_or_json_payload(data: dict) -> dict:
    """
    Maps either legacy form fields (box1..box9) or modern named fields
    into the standardized 9-feature dictionary expected by evaluate_cart_decision_tree.
    """
    return {
        "dependents": data.get("dependents", data.get("box1", 1)),
        "education": data.get("education", data.get("box2", 0)),
        "self_employed": data.get("self_employed", data.get("box3", 2)),
        "applicant_income": data.get("applicant_income", data.get("box4", 5000)),
        "coapplicant_income": data.get("coapplicant_income", data.get("box5", 2000)),
        "loan_amount": data.get("loan_amount", data.get("box6", 150000)),
        "loan_amount_term": data.get("loan_amount_term", data.get("box7", 360)),
        "credit_history": data.get("credit_history", data.get("box8", 1)),
        "property_area": data.get("property_area", data.get("box9", 2)),
    }


try:
    from flask import Flask, jsonify, render_template, request, send_from_directory

    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        if request.is_json:
            payload = parse_box_or_json_payload(request.get_json(silent=True) or {})
            return jsonify(evaluate_cart_decision_tree(payload))
        form_dict = request.form.to_dict()
        payload = parse_box_or_json_payload(form_dict)
        result = evaluate_cart_decision_tree(payload)
        return render_template(
            "index.html",
            loan_status=result["loan_status"],
            initial_result=json.dumps(result),
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        payload = parse_box_or_json_payload(data)
        return jsonify(evaluate_cart_decision_tree(payload))

    @app.route("/api/summary", methods=["GET"])
    def api_summary():
        return jsonify(get_dataset_eda_summary())

    @app.route("/Printable_Documentation.html")
    def printable_doc():
        return send_from_directory(BASE_DIR, "Printable_Documentation.html")

    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    app = None


class FallbackLoanRequestHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, content: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/Launch_Loan_Prediction_App.html"):
            target = os.path.join(BASE_DIR, "index.html")
            with open(target, "rb") as f:
                self._send_bytes(f.read(), "text/html; charset=utf-8")
            return
        if path == "/Printable_Documentation.html":
            target = os.path.join(BASE_DIR, "Printable_Documentation.html")
            if os.path.exists(target):
                with open(target, "rb") as f:
                    self._send_bytes(f.read(), "text/html; charset=utf-8")
                return
        if path == "/api/summary":
            body = json.dumps(get_dataset_eda_summary()).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
            return
        if path.startswith("/static/"):
            rel = path.lstrip("/")
            full_path = os.path.join(BASE_DIR, rel)
            if os.path.isfile(full_path):
                ctype = "text/css; charset=utf-8" if full_path.endswith(".css") else "application/octet-stream"
                with open(full_path, "rb") as f:
                    self._send_bytes(f.read(), ctype)
                return
        self._send_bytes(b"Not Found", "text/plain; charset=utf-8", status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        ctype = self.headers.get("Content-Type", "")
        if "application/json" in ctype:
            try:
                data = json.loads(raw)
            except Exception:
                data = {}
        else:
            parsed = urllib.parse.parse_qs(raw)
            data = {k: v[0] for k, v in parsed.items()}

        payload = parse_box_or_json_payload(data)
        result = evaluate_cart_decision_tree(payload)
        body = json.dumps(result).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")


if __name__ == "__main__":
    print(f"Starting CRED-TREE Loan Approval Underwriting Platform on http://127.0.0.1:{PORT}")
    if HAS_FLASK and app is not None:
        app.run(host="127.0.0.1", port=PORT, debug=False)
    else:
        server = HTTPServer(("127.0.0.1", PORT), FallbackLoanRequestHandler)
        server.serve_forever()
