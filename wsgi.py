"""Production WSGI entrypoint (gunicorn wsgi:app)."""

from gex_core.env_bootstrap import bootstrap_env

bootstrap_env()

from web_app import APP as app
