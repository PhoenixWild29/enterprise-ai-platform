# Container for Amazon Bedrock AgentCore Runtime (ARM64, port 8080, /invocations and /ping).
FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt snowflake-connector-python

COPY *.py ./
RUN useradd --system --uid 10001 agent && chown -R agent /app
USER agent

EXPOSE 8080
CMD ["python", "agentcore_app.py"]
