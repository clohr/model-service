output "ecs_task_definition_arn" {
  value = aws_ecs_task_definition.model_publish_ecs_task_definition.arn
}

output "ecs_task_iam_role_arn" {
  value = aws_iam_role.ecs_task_iam_role.arn
}
