variable "aws_account" {}

variable "environment_name" {}

variable "service_name" {}

variable "vpc_name" {}

variable "ecs_task_iam_role_id" {}

variable "neo4j_bolt_url" {}

variable "neo4j_bolt_user" {
  default = "model_service_user"
}

variable "neo4j_max_connection_lifetime" {
  default = 300
}

variable "log_level" {
  default = "INFO"
}

locals {
  service = element(split("-", var.service_name), 0)
  tier    = element(split("-", var.service_name), 1)
}
