import os

from gex_core import env_bootstrap


def test_load_env_files_does_not_override_existing(monkeypatch, tmp_path):
    env_file = tmp_path / "spx.env"
    env_file.write_text("export UW_API_KEY=file-key\nexport GEX_DEFAULT_TICKERS=SPX\n")
    monkeypatch.setenv("UW_API_KEY", "existing-key")

    loaded = env_bootstrap.load_env_files((env_file,))

    assert loaded == [str(env_file)]
    assert env_bootstrap.uw_api_key() == "existing-key"
    assert os.environ["GEX_DEFAULT_TICKERS"] == "SPX"


def test_load_env_files_reads_export_syntax(monkeypatch, tmp_path):
    monkeypatch.delenv("UW_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('export UW_API_KEY="quoted-key"\n')

    env_bootstrap.load_env_files((env_file,))

    assert env_bootstrap.uw_api_key() == "quoted-key"


def test_uw_api_configured_rejects_blank(monkeypatch):
    monkeypatch.setenv("UW_API_KEY", "   ")
    assert env_bootstrap.uw_api_configured() is False
