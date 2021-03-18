variable "aws_account" {}

variable "environment_name" {}

variable "service_name" {}

variable "vpc_name" {}

variable "image_tag" {}

variable "tier" {}

variable "mount_points" {
  default = ""
}

variable "neo4j_bolt_user" {
  default = "model_publish_user"
}

variable "model_publish_image_url" {
  default = "pennsieve/model-publish"
}

# Fargate task resources

variable "task_memory" {
  default = "2048"
}

variable "task_cpu" {
  default = "1024"
}
