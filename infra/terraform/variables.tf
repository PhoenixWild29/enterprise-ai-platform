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

variable "log_retention_days" {
  type    = number
  default = 90
}
