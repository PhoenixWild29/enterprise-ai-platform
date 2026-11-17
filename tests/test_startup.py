"""Production must refuse to start on development stand-ins."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_main(**env):
    managed = ("IDENTITY_PROVIDER", "CHECKPOINT_BACKEND", "APP_ENV", "AGENT_BACKEND", "WAREHOUSE_BACKEND",
               "RETRIEVER_BACKEND", "CATALOG_PATH", "CORPUS_PATH", "SYNTHESIS_MODE", "SNOWFLAKE_DATABASE")
    full = {k: v for k, v in os.environ.items() if k not in managed} | env
    return subprocess.run([sys.executable, "-c", "import main; print(main.DEV_TOKENS_ENABLED)"],
                          cwd=ROOT, env=full, capture_output=True, text=True)


ENTRA = dict(IDENTITY_PROVIDER="entra", ENTRA_TENANT_ID="t", ENTRA_CLIENT_ID="c", ENTRA_CLIENT_SECRET="s",
             SNOWFLAKE_SCOPE_PREFIX="api://sf")


def _real_backends(tmp_path):
    from schemas import Catalog, DataDomain, TableSpec

    catalog = tmp_path / "catalog.json"
    catalog.write_text(Catalog.from_tables([TableSpec(
        schema_name="OPS", name="PUMP_READINGS", domain=DataDomain.SCADA, columns=("READING_ID", "READING_TS", "VALUE"),
        primary_key=("READING_ID",), time_column="READING_TS", is_sensor=True)], database="PLANT").model_dump_json())
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"passage_id": "d1", "source": "doc://x", "text": "Pump seals are replaced yearly.", "domain": "MAINTENANCE"}\n')
    return dict(AGENT_BACKEND="llm", AGENT_MODEL="some-model", WAREHOUSE_BACKEND="snowflake", SNOWFLAKE_ACCOUNT="acme",
                SNOWFLAKE_WAREHOUSE="AI_WH", SNOWFLAKE_DATABASE="PLANT", RETRIEVER_BACKEND="vector",
                CORPUS_PATH=str(corpus), CATALOG_PATH=str(catalog),
                CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=str(tmp_path / "cp.sqlite"))


def test_production_refuses_every_development_stand_in():
    r = _import_main(APP_ENV="production", CHECKPOINT_BACKEND="memory")
    assert r.returncode != 0
    for needle in ("AGENT_BACKEND=llm", "WAREHOUSE_BACKEND=snowflake", "RETRIEVER_BACKEND=vector", "CATALOG_PATH",
                   "IDENTITY_PROVIDER=entra", "durable CHECKPOINT_BACKEND"):
        assert needle in r.stderr, needle


def test_production_refuses_a_single_remaining_stand_in(tmp_path):
    real = _real_backends(tmp_path)
    r = _import_main(APP_ENV="production", **ENTRA, **(real | {"WAREHOUSE_BACKEND": "simulated"}))
    assert r.returncode != 0 and "WAREHOUSE_BACKEND=snowflake" in r.stderr and "AGENT_BACKEND=llm" not in r.stderr
    r = _import_main(APP_ENV="production", **real)  # everything real except identity
    assert r.returncode != 0 and "IDENTITY_PROVIDER=entra" in r.stderr


def test_production_refuses_sample_corpus_demo_catalog_and_model_written_answers(tmp_path):
    from gates import DEFAULT_CATALOG

    real = _real_backends(tmp_path)
    demo = tmp_path / "demo.json"
    demo.write_text(DEFAULT_CATALOG.model_dump_json())
    for override, needle in (({"CORPUS_PATH": ""}, "CORPUS_PATH"),
                             ({"SYNTHESIS_MODE": "abstractive"}, "SYNTHESIS_MODE=extractive"),
                             ({"CATALOG_PATH": str(demo), "SNOWFLAKE_DATABASE": "LAKEHOUSE"}, "demo catalog"),
                             ({"SNOWFLAKE_DATABASE": "SOMEWHERE_ELSE"}, "SNOWFLAKE_DATABASE to match")):
        r = _import_main(APP_ENV="production", **ENTRA, **(real | override))
        assert r.returncode != 0 and needle in r.stderr, (override, r.stderr[-300:])


def test_production_starts_on_real_backends_with_dev_tokens_off(tmp_path):
    r = _import_main(APP_ENV="production", **ENTRA, **_real_backends(tmp_path))
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr


def test_environment_names_fail_closed():
    for name in ("Production", "PRODUCTION", " production ", "prod", "staging", "live"):
        r = _import_main(APP_ENV=name, CHECKPOINT_BACKEND="memory")
        assert r.returncode != 0 and "treated as production" in r.stderr, name
    for name in ("dev", "Development", "local", "test"):
        r = _import_main(APP_ENV=name, CHECKPOINT_BACKEND="memory")
        assert r.returncode == 0 and r.stdout.strip() == "True", (name, r.stderr)


def test_dev_defaults_start():
    r = _import_main(CHECKPOINT_BACKEND="memory")
    assert r.returncode == 0 and r.stdout.strip() == "True", r.stderr
