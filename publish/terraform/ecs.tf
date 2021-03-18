data "template_file" "task_definition" {
  template = file("${path.module}/task-definition.json")

  vars = {
    aws_region                = data.aws_region.current_region.name
    cloudwatch_log_group_name = data.terraform_remote_state.ecs_cluster.outputs.cloudwatch_log_group_name
    image_tag                 = var.image_tag
    environment_name          = var.environment_name
    service_name              = var.service_name
    tier                      = var.tier
    mount_points              = var.mount_points
    docker_hub_credentials    = data.terraform_remote_state.platform_infrastructure.outputs.docker_hub_credentials_arn

    s3_publish_bucket   = data.terraform_remote_state.platform_infrastructure.outputs.discover_publish_bucket_id

    model_publish_image_url     = var.model_publish_image_url
    model_publish_stream_prefix = "${var.environment_name}-${var.service_name}-${var.tier}"
    neo4j_bolt_url              = data.terraform_remote_state.model_service.outputs.neo4j_bolt_url
    neo4j_bolt_user_arn         = aws_ssm_parameter.neo4j_bolt_user.arn
    neo4j_bolt_password_arn     = aws_ssm_parameter.neo4j_bolt_password.arn
  }
}

# Create ECS task - model-publish
resource "aws_ecs_task_definition" "model_publish_ecs_task_definition" {
  family                   = "${var.environment_name}-${var.service_name}-${var.tier}-${data.terraform_remote_state.vpc.outputs.aws_region_shortname}"
  network_mode             = "awsvpc"
  container_definitions    = data.template_file.task_definition.rendered
  task_role_arn            = aws_iam_role.ecs_task_iam_role.arn
  execution_role_arn       = aws_iam_role.ecs_task_iam_role.arn
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory

  depends_on = [
    data.template_file.task_definition,
    aws_iam_role.ecs_task_iam_role,
  ]
}
