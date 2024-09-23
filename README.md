# Personal Website

Simple [HTML5Up](https://html5up.net/) based website that links out to my blog and 
various other bits. 

Backend is made with [Flask](https://flask.palletsprojects.com/en/3.0.x/).

## Setup

The frontend is deployed with [Github Pages](https://pages.github.com/) and the 
backend is deployed using [Render](https://render.com/).  To run this locally, 

```bash
pip install poetry
poetry install
export EMAIL_ADDRESS=YOUR_EMAIL_ADDRESS
export MAIL_ADDRESS_PASSWORD=YOUR_GMAIL_APP_PASSWORD
export FLASK_SECRET_KEY=FLASK_SECRET_KEY
poetry run python main.py --port 8000
```

### Maintainers

* westford14
