"""Production WSGI entrypoint (gunicorn wsgi:app)."""

from gex_core.env_bootstrap import load_env_files, sync_env_files_from_process

sync_env_files_from_process()
load_env_files()

from web_app import APP as app
