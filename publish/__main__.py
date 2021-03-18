import argparse
import logging
from os import environ
from typing import Any, Callable

import boto3  # type: ignore

from server.db import PartitionedDatabase
from server.logging import configure_logging
from server.models import (
    DatasetId,
    DatasetNodeId,
    OrganizationId,
    OrganizationNodeId,
    UserNodeId,
)

from .config import PublishConfig
from .models import ModelService
from .publish import publish_dataset, read_file_manifests, write_graph_manifests

log = logging.getLogger(__name__)


class EnvFallbackAction(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs: Any):
        if not default and envvar:
            if envvar in environ:
                default = environ[envvar]
        if required and default:
            required = False
        super(EnvFallbackAction, self).__init__(
            default=default, required=required, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


def env_action(var: str):
    def make(**kwargs):
        return EnvFallbackAction(var, **kwargs)

    return make


parser = argparse.ArgumentParser(
    description="Publish a dataset to the Pennsieve Discover platform."
)
parser.add_argument(
    "--organization-id",
    type=int,
    required=True,
    help="Organization integer ID",
    action=env_action("ORGANIZATION_ID"),
)
parser.add_argument(
    "--organization-node-id",
    type=str,
    required=True,
    help="Organization node ID",
    action=env_action("ORGANIZATION_NODE_ID"),
)
parser.add_argument(
    "--dataset-id",
    type=int,
    required=True,
    help="Dataset integer ID",
    action=env_action("DATASET_ID"),
)
parser.add_argument(
    "--dataset-node-id",
    type=str,
    required=True,
    help="Dataset node ID",
    action=env_action("DATASET_NODE_ID"),
)
parser.add_argument(
    "--user-id",
    type=str,
    required=True,
    help="User integer ID",
    action=env_action("USER_ID"),
)
parser.add_argument(
    "--user-node-id",
    type=str,
    required=True,
    help="User node ID",
    action=env_action("USER_NODE_ID"),
)
parser.add_argument(
    "--s3-publish-key",
    type=str,
    help="AWS S3 publishing key",
    action=env_action("S3_PUBLISH_KEY"),
)
parser.add_argument(
    "--s3-bucket",
    type=str,
    help="AWS S3 target bucket, either for embargoed or published datasets",
    action=env_action("S3_BUCKET"),
)


if __name__ == "__main__":
    configure_logging("INFO")

    args = parser.parse_args()
    db = PartitionedDatabase.get_from_env(
        organization_id=OrganizationId(args.organization_id),
        dataset_id=DatasetId(args.dataset_id),
        user_id=UserNodeId(args.user_node_id),
        organization_node_id=OrganizationNodeId(args.organization_node_id),
        dataset_node_id=DatasetNodeId(args.dataset_node_id),
    )

    s3 = boto3.client("s3", region_name="us-east-1")

    config = PublishConfig(s3_publish_key=args.s3_publish_key, s3_bucket=args.s3_bucket)

    file_manifests = read_file_manifests(s3, config)

    graph_manifests = publish_dataset(db, s3, config, file_manifests=file_manifests)

    write_graph_manifests(s3, config, graph_manifests)
