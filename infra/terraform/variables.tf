variable "region" {
  type        = string
  description = "AWS region for the agent runtime and Bedrock."
}

variable "name" {
  type        = string
  default     = "enterprise_ai_platform"
  description = "Name of the AgentCore runtime."
}

variable "image_tag" {
  type        = string
  description = "Tag of the container image in ECR to run."
}

variable "subnet_ids" {
  type        = list(string)
  description = "Private subnets for the runtime. Model calls and data stay inside this VPC."
}

variable "security_group_ids" {
  type        = list(string)
  description = "Security groups for the runtime."
}

variable "entra_tenant_id" {
  type        = string
  description = "Microsoft Entra tenant that issues user tokens."
}

variable "api_audience" {
  type        = string
  description = "Audience of inbound user tokens (the platform API's app registration)."
}

variable "allowed_client_ids" {
  type        = list(string)
  description = "Client applications (people's apps and scheduled jobs) allowed to call the platform, matched against the token's azp."
}

variable "bedrock_model_arns" {
  type        = list(string)
  description = "Foundation models and inference profiles the agent may invoke."
}

variable "environment" {
  type        = map(string)
  description = "Further non-secret settings for the container. Must include the Snowflake account and warehouse."

  validation {
    condition     = alltrue([for key in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_WAREHOUSE"] : contains(keys(var.environment), key)])
    error_message = "environment must set SNOWFLAKE_ACCOUNT and SNOWFLAKE_WAREHOUSE."
  }
}

variable "memory_event_expiry_days" {
  type        = number
  default     = 180
  description = "Retention for AgentCore Memory events."
}

variable "entra_client_id" {
  type        = string
  description = "Application (client) id of the platform API's Entra app registration."
}

variable "snowflake_scope_prefix" {
  type        = string
  description = "Snowflake OAuth resource, for example api://<snowflake app id>."
}

variable "agent_model" {
  type        = string
  description = "Bedrock model id or inference profile the agent runs on. Must be covered by bedrock_model_arns."
}

variable "obo_credential_provider_name" {
  type        = string
  description = "AgentCore Identity credential provider used for the on-behalf-of exchange."
}

variable "m2m_credential_provider_name" {
  type        = string
  description = "AgentCore Identity credential provider used for scheduled jobs."
}

variable "identity_resource_arns" {
  type        = list(string)
  description = "ARNs of the workload identity, token vault and credential providers this agent may use."
}

variable "credential_secret_arns" {
  type        = list(string)
  description = "Secrets Manager ARNs holding the credential providers' client secrets."
}

variable "service_principals_json" {
  type        = string
  default     = "[]"
  description = "Registered scheduled jobs: client id, service account, execution role and allowed DAG ids. Holds no secrets."
}

variable "catalog_path" {
  type        = string
  default     = "/app/deploy/catalog.json"
  description = "Path in the image of the catalog exported from Snowflake (see the Dockerfile)."
}

variable "retriever_backend" {
  type        = string
  description = "Document retrieval backend. snowflake also needs PASSAGES_TABLE and SNOWFLAKE_CONNECTION_NAME in var.environment; vector and hybrid need CORPUS_PATH."

  validation {
    condition     = contains(["snowflake", "vector", "hybrid"], var.retriever_backend)
    error_message = "retriever_backend must be snowflake, vector or hybrid."
  }
}

variable "kms_key_arn" {
  type        = string
  default     = null
  description = "Customer-managed KMS key for the image repository. AWS-managed encryption when null."
}
