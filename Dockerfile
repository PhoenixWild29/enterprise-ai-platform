# Container for Amazon Bedrock AgentCore Runtime (ARM64, port 8080, /invocations and /ping).
#
# Before a real release: pin the base image by digest and install from a lock
# file with hashes (pip-compile --generate-hashes). Neither is done here.
FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Runtime dependencies only; test tools live in requirements-dev.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "snowflake-connector-python>=3.12,<5"

COPY *.py ./
# deploy/ holds what production needs besides code: the catalog exported from
# Snowflake (python snowflake_backend.py export-catalog --overlay governance.json --out deploy/catalog.json)
# and, for RETRIEVER_BACKEND=vector or hybrid, the document corpus.
COPY deploy/ ./deploy/
RUN useradd --system --uid 10001 agent && chown -R agent /app
USER agent

EXPOSE 8080
CMD ["python", "agentcore_app.py"]
