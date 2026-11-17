# Amazon Bedrock AgentCore deployment of the platform.
#
# NOT YET APPLIED OR VALIDATED. Written against the AWS provider documentation
# for aws_bedrockagentcore_agent_runtime and aws_bedrockagentcore_memory.
# Run `terraform validate` and `terraform plan` before relying on it, and check
# the IAM action names in the execution policy against the current service
# authorization reference.

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  # AgentCore Runtime creates its own log groups, named from the generated runtime id.
  runtime_log_groups = "arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"
}

resource "aws_ecr_repository" "agent" {
  name                 = replace(var.name, "_", "-")
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = var.kms_key_arn == null ? "AES256" : "KMS"
    kms_key         = var.kms_key_arn
  }
}

data "aws_iam_policy_document" "assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:bedrock-agentcore:${var.region}:${local.account_id}:*"]
    }
  }
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "PullImageAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid       = "PullImage"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
    resources = [aws_ecr_repository.agent.arn]
  }

  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:DescribeLogStreams", "logs:PutLogEvents"]
    resources = [local.runtime_log_groups, "${local.runtime_log_groups}:*"]
  }

  statement {
    sid       = "Traces"
    actions   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
    resources = ["*"]
  }

  # Least privilege: only the named models, nothing else in Bedrock.
  statement {
    sid       = "InvokeApprovedModels"
    actions   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = var.bedrock_model_arns
  }

  # AgentCore Identity: exchange the caller's workload token for a Snowflake token.
  # Scoped to this agent's workload identity, its token vault and the two
  # credential providers it uses; no wildcard.
  statement {
    sid = "WorkloadIdentityTokens"
    actions = [
      "bedrock-agentcore:GetWorkloadAccessToken",
      "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
      "bedrock-agentcore:GetResourceOauth2Token",
    ]
    resources = var.identity_resource_arns
  }

  # The credential providers keep their client secrets in Secrets Manager.
  statement {
    sid       = "CredentialProviderSecrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = var.credential_secret_arns
  }

  dynamic "statement" {
    for_each = var.kms_key_arn == null ? [] : [var.kms_key_arn]
    content {
      sid       = "DecryptImage"
      actions   = ["kms:Decrypt"]
      resources = [statement.value]
    }
  }

  # Memory is append-only from the agent: forgetting an entry writes a tombstone event.
  statement {
    sid       = "AgentMemory"
    actions   = ["bedrock-agentcore:CreateEvent", "bedrock-agentcore:ListEvents", "bedrock-agentcore:GetEvent"]
    resources = [aws_bedrockagentcore_memory.agent.arn]
  }
}

resource "aws_iam_role" "agent" {
  name               = "${replace(var.name, "_", "-")}-runtime"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy" "agent" {
  role   = aws_iam_role.agent.id
  policy = data.aws_iam_policy_document.execution.json
}

resource "aws_bedrockagentcore_memory" "agent" {
  name                  = "${var.name}_memory"
  description           = "User-scoped preferences. Shared memory is review-gated in the application."
  event_expiry_duration = var.memory_event_expiry_days
}

resource "aws_bedrockagentcore_agent_runtime" "agent" {
  agent_runtime_name = var.name
  description        = "Gated enterprise agent: Strands on AgentCore Runtime"
  role_arn           = aws_iam_role.agent.arn

  agent_runtime_artifact {
    container_configuration {
      container_uri = "${aws_ecr_repository.agent.repository_url}:${var.image_tag}"
    }
  }

  # Private networking: the runtime reaches Bedrock, Snowflake and ECR through the VPC.
  # That means VPC endpoints (or NAT) for Bedrock, AgentCore, ECR and CloudWatch Logs, a route
  # to Snowflake, and HTTPS egress to login.microsoftonline.com for Entra signing keys.
  network_configuration {
    network_mode = "VPC"

    network_mode_config {
      subnets         = var.subnet_ids
      security_groups = var.security_group_ids
    }
  }

  # Inbound tokens are checked by the runtime before the container sees them, and
  # checked again in the application. People and scheduled jobs both call with an
  # Entra token for the API audience: a job signs in with its own client credentials
  # and its client id must be in allowed_client_ids. The application reads the job's
  # identity from that token (needs the optional idtyp claim on the app registration).
  authorizer_configuration {
    custom_jwt_authorizer {
      discovery_url    = "https://login.microsoftonline.com/${var.entra_tenant_id}/v2.0/.well-known/openid-configuration"
      allowed_audience = [var.api_audience]
      # allowed_clients is left out on purpose: it has not been confirmed which claim the
      # runtime matches it against, and Entra v2 tokens carry the client in azp. The
      # application enforces the client allow-list itself (ENTRA_ALLOWED_CLIENT_IDS).
    }
  }

  # The application needs the caller's token to map roles and log the principal.
  request_header_configuration {
    request_header_allowlist = ["Authorization"]
  }

  # No secrets here: environment variables are stored in plain text and in state.
  # The fixed settings come last so var.environment cannot override them.
  environment_variables = merge(
    var.environment,
    {
      APP_ENV                 = "production"
      IDENTITY_PROVIDER       = "agentcore"
      WAREHOUSE_BACKEND       = "snowflake"
      RETRIEVER_BACKEND       = var.retriever_backend
      CATALOG_PATH            = var.catalog_path
      AWS_REGION              = var.region
      AGENT_MODEL             = var.agent_model
      ENTRA_TENANT_ID         = var.entra_tenant_id
      ENTRA_CLIENT_ID         = var.entra_client_id
      ENTRA_API_AUDIENCE      = var.api_audience
      ENTRA_ALLOWED_CLIENT_IDS = join(",", var.allowed_client_ids)
      SNOWFLAKE_SCOPE_PREFIX  = var.snowflake_scope_prefix
      AGENTCORE_OBO_PROVIDER  = var.obo_credential_provider_name
      AGENTCORE_M2M_PROVIDER  = var.m2m_credential_provider_name
      SERVICE_PRINCIPALS_JSON = var.service_principals_json
      AGENTCORE_MEMORY_ID     = aws_bedrockagentcore_memory.agent.id
    },
  )

  depends_on = [aws_iam_role_policy.agent]
}
