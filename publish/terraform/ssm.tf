resource "aws_ssm_parameter" "neo4j_bolt_url" {
  name  = "/${var.environment_name}/${var.service_name}-${var.tier}/neo4j-bolt-url"
  type  = "String"
  value = data.terraform_remote_state.model_service.outputs.neo4j_bolt_url
  # value = "bolt://${data.terraform_remote_state.neo4j_infrastructure.outputs.internal_fqdn}:7687"
}

resource "aws_ssm_parameter" "neo4j_bolt_user" {
  name  = "/${var.environment_name}/${var.service_name}-${var.tier}/neo4j-bolt-user"
  type  = "String"
  value = var.neo4j_bolt_user
}

resource "aws_ssm_parameter" "neo4j_bolt_password" {
  name      = "/${var.environment_name}/${var.service_name}-${var.tier}/neo4j-bolt-password"
  type      = "SecureString"
  value     = "dummy"
  overwrite = false

  lifecycle {
    ignore_changes = [value]
  }
}
