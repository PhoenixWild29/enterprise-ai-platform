Deployment inputs copied into the container image at /app/deploy.

catalog.json   The governed catalog exported from Snowflake:
                   python snowflake_backend.py export-catalog --overlay governance.json --out deploy/catalog.json
               Point CATALOG_PATH at /app/deploy/catalog.json.
corpus.jsonl   Only for RETRIEVER_BACKEND=vector or hybrid: one passage per line
               with passage_id, source, text and domain. Point CORPUS_PATH at it.

Neither file is committed. Without a catalog the production start-up check fails.
