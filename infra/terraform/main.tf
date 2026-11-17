# Amazon Bedrock AgentCore deployment of the platform.
#
# NOT YET APPLIED OR VALIDATED. Written against the AWS provider documentation
# for aws_bedrockagentcore_agent_runtime and aws_bedrockagentcore_memory.
# Run `terraform validate` and `terraform plan` before relying on it, and check
# the IAM action names in the execution policy against the current service
# authorization reference.

data "aws_caller_identity" "current" {}

resource "aws_ecr_repository" "agent" {
  name                 = replace(var.name, "_", "-")
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_cloudwatch_log_group" "agent" {
  name              = "/aws/bedrock-agentcore/runtimes/${var.name}"
  retention_in_days = var.log_retention_days
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
      values   = [data.aws_caller_identity.current.account_id]
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
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.agent.arn}:*"]
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
  statement {
    sid = "WorkloadIdentityTokens"
    actions = [
      "bedrock-agentcore:GetWorkloadAccessToken",
      "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
      "bedrock-agentcore:GetResourceOauth2Token",
    ]
    resources = ["*"]
  }

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
  network_configuration {
    network_mode = "VPC"

    network_mode_config {
      subnets         = var.subnet_ids
      security_groups = var.security_group_ids
    }
  }

  # Inbound user tokens are checked by the runtime before the container sees them,
  # and checked again in the application.
  authorizer_configuration {
    custom_jwt_authorizer {
      discovery_url    = "https://login.microsoftonline.com/${var.entra_tenant_id}/v2.0/.well-known/openid-configuration"
      allowed_audience = [var.api_audience]
      allowed_clients  = var.allowed_client_ids
    }
  }

  # The application needs the caller's token to map roles and log the principal.
  request_header_configuration {
    request_header_allowlist = ["Authorization"]
  }

  environment_variables = merge(
    {
      APP_ENV              = "production"
      IDENTITY_PROVIDER    = "agentcore"
      WAREHOUSE_BACKEND    = "snowflake"
      AWS_REGION           = var.region
      ENTRA_TENANT_ID      = var.entra_tenant_id
      ENTRA_API_AUDIENCE   = var.api_audience
      AGENTCORE_MEMORY_ID  = aws_bedrockagentcore_memory.agent.id
    },
    var.environment,
  )

  depends_on = [aws_iam_role_policy.agent]
}
