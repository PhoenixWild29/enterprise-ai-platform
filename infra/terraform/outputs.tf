output "agent_runtime_arn" {
  value = aws_bedrockagentcore_agent_runtime.agent.agent_runtime_arn
}

output "ecr_repository_url" {
  value = aws_ecr_repository.agent.repository_url
}

output "memory_id" {
  value = aws_bedrockagentcore_memory.agent.id
}
