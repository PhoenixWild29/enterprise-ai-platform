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
  description = "Client applications allowed to call the runtime."
}

variable "bedrock_model_arns" {
  type        = list(string)
  description = "Foundation models and inference profiles the agent may invoke."
}

variable "environment" {
  type        = map(string)
  default     = {}
  description = "Extra environment variables for the container (catalog path, Snowflake account, model ids)."
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
  default     = "snowflake"
  description = "Document retrieval backend."
}

variable "kms_key_arn" {
  type        = string
  default     = null
  description = "Customer-managed KMS key for the image repository. AWS-managed encryption when null."
}
