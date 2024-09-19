"""Simple webserver.

Simple webserver backend to allow for sending of emails and other
tasks that cannot be handled directly by the simple website.
"""

import argparse
import os

from flask import Flask, request  # noqa: I201
from flask_mail import Mail, Message  # noqa: I201
from gevent.pywsgi import WSGIServer  # noqa: I201
from markupsafe import escape  # noqa: I201

app = Flask(__name__)
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USERNAME"] = os.environ["EMAIL_ADDRESS"]
app.config["MAIL_PASSWORD"] = os.environ["EMAIL_ADDRESS_PASSWORD"]
app.config["MAIL_USE_TLS"] = False
app.config["MAIL_USE_SSL"] = True
mail = Mail(app)


@app.route("/healthz", methods=["GET"])
def health_check() -> str:
    """Run a simple health check."""
    return "healthy"


@app.route("/submit", methods=["POST"])
def submit() -> None:
    """Gather the data from the submitted email form."""
    name = escape(request.form["name"])
    email = escape(request.form["email"])
    message = escape(request.form["message"])
    recipient: str = os.environ["EMAIL_ADDRESS"]

    mail_message = Message(
        subject="Inquiry",
        sender=email,
        recipients=[recipient],
    )
    mail_message.body = f"From: {name}\n{message}"
    mail.send(mail_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", default=8000, help="the port to use")
    args = parser.parse_args()

    server = WSGIServer(("0.0.0.0", int(args.port)), app)
    server.serve_forever()
