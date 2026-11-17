"""Production must refuse to start on development stand-ins."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_main(**env):
    full = {k: v for k, v in os.environ.items() if k not in ("IDENTITY_PROVIDER", "CHECKPOINT_BACKEND", "APP_ENV")} | env
    return subprocess.run([sys.executable, "-c", "import main; print(main.DEV_TOKENS_ENABLED)"],
                          cwd=ROOT, env=full, capture_output=True, text=True)


def test_production_refuses_simulated_identity_provider():
    r = _import_main(APP_ENV="production", CHECKPOINT_BACKEND="memory")
    assert r.returncode != 0 and "requires IDENTITY_PROVIDER=entra" in r.stderr


def test_production_refuses_memory_checkpointer_and_disables_dev_tokens(tmp_path):
    entra = dict(APP_ENV="production", IDENTITY_PROVIDER="entra", ENTRA_TENANT_ID="t", ENTRA_CLIENT_ID="c",
                 ENTRA_CLIENT_SECRET="s", SNOWFLAKE_SCOPE_PREFIX="api://sf")
    r = _import_main(**entra, CHECKPOINT_BACKEND="memory")
    assert r.returncode != 0 and "durable CHECKPOINT_BACKEND" in r.stderr
    r = _import_main(**entra, CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=str(tmp_path / "cp.sqlite"))
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr


def test_dev_defaults_start():
    r = _import_main(CHECKPOINT_BACKEND="memory")
    assert r.returncode == 0 and r.stdout.strip() == "True", r.stderr
