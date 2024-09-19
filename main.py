"""Simple webserver.

Simple webserver backend to allow for sending of emails and other
tasks that cannot be handled directly by the simple website.
"""

import argparse
import os

from flask import Flask, flash, redirect, request  # noqa: I201
from flask_mail import Mail, Message  # noqa: I201
from gevent.pywsgi import WSGIServer  # noqa: I201
from markupsafe import escape  # noqa: I201
from werkzeug.wrappers.response import Response  # noqa: I201

from logger import get_logger  # noqa: I100

logger = get_logger(__name__)
app = Flask(__name__)
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USERNAME"] = os.environ["EMAIL_ADDRESS"]
app.config["MAIL_PASSWORD"] = escape(
    os.environ["EMAIL_ADDRESS_PASSWORD"]  # noqa
).striptags()  # noqa
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["TESTING"] = False
app.config["MAIL_SUPPRESS_SEND"] = False
app.config["MAIL_DEBUG"] = True
app.config["MAIL_ASCII_ATTACHMENTS"] = False
mail = Mail(app)


@app.route("/healthz", methods=["GET"])
def health_check() -> str:
    """Run a simple health check."""
    return "healthy"


@app.route("/submit", methods=["POST"])
def submit() -> Response:
    """Gather the data from the submitted email form."""
    name = escape(request.form["name"]).striptags()
    email = escape(request.form["email"]).striptags()
    message = escape(request.form["message"]).striptags()

    recipient: str = os.environ["EMAIL_ADDRESS"]
    logger.info(f"received request: {name} - {email} - {message}")

    mail_message = Message(
        subject="Inquiry",
        sender=email,
        recipients=[recipient],
    )
    mail_message.body = f"From: {name}\n{message}"
    try:
        mail.send(mail_message)
        logger.info("sent message")
        flash("Inquiry successfully sent", "success")
    except Exception as e:
        logger.error(f"could not process mail send {e}")
        flash("Error on submission -- will be looked into :)", "error")
    return redirect("https://westford14.github.io/index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", default=8000, help="the port to use")
    args = parser.parse_args()
    logger.info(f"starting WSGI server on port: {args.port}")

    app.secret_key = os.environ["FLASK_SECRET_KEY"]

    server = WSGIServer(("0.0.0.0", int(args.port)), app)
    logger.info("starting server")
    server.serve_forever()
