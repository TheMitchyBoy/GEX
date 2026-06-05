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


def test_sync_env_files_from_process_writes_missing_env(monkeypatch, tmp_path):
    monkeypatch.setenv("UW_API_KEY", "persisted-key")
    target = tmp_path / ".env"

    written = env_bootstrap.sync_env_files_from_process(target)

    assert written == [str(target)]
    assert target.read_text(encoding="utf-8") == "UW_API_KEY=persisted-key\n"
    assert (target.stat().st_mode & 0o777) == 0o600


def test_load_env_files_overrides_blank_env_placeholder(monkeypatch, tmp_path):
    monkeypatch.setenv("UW_API_KEY", "")
    env_file = tmp_path / ".env"
    env_file.write_text("UW_API_KEY=file-key\n")

    env_bootstrap.load_env_files((env_file,))

    assert env_bootstrap.uw_api_key() == "file-key"


def test_blank_env_placeholder_blocks_until_file_load(monkeypatch, tmp_path):
    monkeypatch.setenv("UW_API_KEY", "")
    env_file = tmp_path / ".env"
    env_file.write_text("UW_API_KEY=file-key\n")

    assert env_bootstrap.uw_api_configured() is False
    env_bootstrap.load_env_files((env_file,))
    assert env_bootstrap.uw_api_configured() is True


def test_bootstrap_env_syncs_then_loads(monkeypatch, tmp_path):
    monkeypatch.delenv("UW_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(env_bootstrap, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("UW_API_KEY", "bootstrap-key")

    loaded = env_bootstrap.bootstrap_env((env_file,))

    assert loaded == [str(env_file)]
    assert env_bootstrap.uw_api_key() == "bootstrap-key"
    assert env_file.read_text(encoding="utf-8") == "UW_API_KEY=bootstrap-key\n"


def test_sync_env_files_from_process_skips_existing_key(monkeypatch, tmp_path):
    monkeypatch.setenv("UW_API_KEY", "new-key")
    target = tmp_path / ".env"
    target.write_text("UW_API_KEY=existing-key\n", encoding="utf-8")

    assert env_bootstrap.sync_env_files_from_process(target) == []
    assert target.read_text(encoding="utf-8") == "UW_API_KEY=existing-key\n"
