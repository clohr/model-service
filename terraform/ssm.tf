// NEO4J CONFIGURATION

resource "aws_ssm_parameter" "neo4j_bolt_url" {
  name  = "/${var.environment_name}/${var.service_name}/neo4j-bolt-url"
  type  = "String"
  value = var.neo4j_bolt_url
  # value = "bolt://${data.terraform_remote_state.neo4j_infrastructure.outputs.internal_fqdn}:7687"
}

resource "aws_ssm_parameter" "neo4j_bolt_user" {
  name  = "/${var.environment_name}/${var.service_name}/neo4j-bolt-user"
  type  = "String"
  value = var.neo4j_bolt_user
}

resource "aws_ssm_parameter" "neo4j_bolt_password" {
  name      = "/${var.environment_name}/${var.service_name}/neo4j-bolt-password"
  overwrite = false
  type      = "SecureString"
  value     = "dummy"

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "neo4j_max_connection_lifetime" {
  name  = "/${var.environment_name}/${var.service_name}/neo4j-max-connection-lifetime"
  type  = "String"
  value = var.neo4j_max_connection_lifetime
}

// JWT CONFIGURATION

resource "aws_ssm_parameter" "jwt_secret_key" {
  name      = "/${var.environment_name}/${var.service_name}/jwt-secret-key"
  overwrite = false
  type      = "SecureString"
  value     = "dummy"

  lifecycle {
    ignore_changes = [value]
  }
}

// NEW RELIC

resource "aws_ssm_parameter" "new_relic_labels" {
  name  = "/${var.environment_name}/${var.service_name}/new-relic-labels"
  type  = "String"
  value = "Environment:${var.environment_name};Service:${local.service};Tier:${local.tier}"
}

resource "aws_ssm_parameter" "new_relic_app_name" {
  name  = "/${var.environment_name}/${var.service_name}/new-relic-app-name"
  type  = "String"
  value = "${var.environment_name}-${var.service_name}"
}

resource "aws_ssm_parameter" "new_relic_license_key" {
  name      = "/${var.environment_name}/${var.service_name}/new-relic-license-key"
  overwrite = false
  type      = "SecureString"
  value     = "dummy"

  lifecycle {
    ignore_changes = [value]
  }
}

// LOGGING

resource "aws_ssm_parameter" "log_level" {
  name  = "/${var.environment_name}/${var.service_name}/log-level"
  type  = "String"
  value = var.log_level
}

# // PENNSIEVE SERVICES

resource "aws_ssm_parameter" "pennsieve_api_host" {
  name  = "/${var.environment_name}/${var.service_name}/pennsieve-api-host"
  type  = "String"
  value = "${var.environment_name}-api-${data.terraform_remote_state.region.outputs.aws_region_shortname}.${data.terraform_remote_state.account.outputs.domain_name}"
}

// GATEWAY

resource "aws_ssm_parameter" "gateway_internal_host" {
  name  = "/${var.environment_name}/${var.service_name}/gateway-internal-host"
  type  = "String"
  value = data.terraform_remote_state.gateway.outputs.internal_fqdn
}

// JOBS QUEUE

resource "aws_ssm_parameter" "jobs_sqs_queue_id" {
  name  = "/${var.environment_name}/${var.service_name}/jobs-sqs-queue-id"
  type  = "String"
  value = data.terraform_remote_state.platform_infrastructure.outputs.jobs_queue_id
}
