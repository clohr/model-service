# ECS IAM Role
resource "aws_iam_role" "ecs_task_iam_role" {
  name = "${var.environment_name}-${var.service_name}-${var.tier}-ecs-task-role-${data.terraform_remote_state.vpc.outputs.aws_region_shortname}"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement":
  [
    {
      "Sid": "",
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

# Create IAM Policy
resource "aws_iam_policy" "ecs_task_iam_policy" {
  name   = "${var.environment_name}-${var.service_name}-${var.tier}-policy-${data.terraform_remote_state.vpc.outputs.aws_region_shortname}"
  path   = "/"
  policy = data.aws_iam_policy_document.ecs_task_iam_policy_document.json
}

# Attach IAM Policy
resource "aws_iam_role_policy_attachment" "ecs_task_iam_policy_attachment" {
  role       = aws_iam_role.ecs_task_iam_role.name
  policy_arn = aws_iam_policy.ecs_task_iam_policy.arn
}

# ECS task IAM Policy Document
data "aws_iam_policy_document" "ecs_task_iam_policy_document" {
  statement {
    sid    = "CloudwatchLogPermissions"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutDestination",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]

    resources = ["*"]
  }

  statement {
    sid    = "SSMGetParameters"
    effect = "Allow"

    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
      "ssm:GetParameterHistory",
      "ssm:GetParametersByPath",
    ]

    resources = ["arn:aws:ssm:${data.aws_region.current_region.name}:${data.aws_caller_identity.current.account_id}:parameter/${var.environment_name}/${var.service_name}-${var.tier}/*"]
  }

  statement {
    sid    = "SecretsManagerPermissions"
    effect = "Allow"

    actions = [
      "kms:Decrypt",
      "secretsmanager:GetSecretValue",
    ]

    resources = [
      data.terraform_remote_state.account.docker_hub_credentials_arn,
      data.aws_kms_key.ssm_kms_key.arn,
    ]
  }

  statement {
    sid       = "KMSDecryptSSMSecrets"
    effect    = "Allow"
    actions   = ["kms:*"]
    resources = ["arn:aws:kms:${data.aws_region.current_region.name}:${data.aws_caller_identity.current.account_id}:key/alias/aws/ssm"]
  }

  statement {
    sid     = "S3PublishBucketPermissions"
    effect  = "Allow"
    actions = [
       "s3:GetObject",
       "s3:PutObject",
       "s3:ListBucket"
    ]

    resources = [
      data.terraform_remote_state.platform_infrastructure.outputs.discover_publish_bucket_arn,
      "${data.terraform_remote_state.platform_infrastructure.outputs.discover_publish_bucket_arn}/*",
      data.terraform_remote_state.platform_infrastructure.outputs.discover_embargo_bucket_arn,
      "${data.terraform_remote_state.platform_infrastructure.outputs.discover_embargo_bucket_arn}/*"
    ]
  }

  statement {
    sid    = "EC2Permissions"
    effect = "Allow"
    actions = [
      "ec2:DeleteNetworkInterface",
      "ec2:CreateNetworkInterface",
      "ec2:AttachNetworkInterface",
      "ec2:DescribeNetworkInterfaces",
    ]

    resources = ["*"]
  }
}
