import argparse
import os
import sys

from server.config import Config
from server.db import Database

from .import_to_neo4j import load, parse_truthy

S3_DATA_BUCKET = os.environ.get("S3_DATA_BUCKET", None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="neo4j data loader")
    parser.add_argument(
        "--bucket",
        type=str,
        help="The S3 bucket containing data files. Can be overridden by S3_DATA_BUCKET",
        default=S3_DATA_BUCKET,
        required=False,
    )
    parser.add_argument(
        "--dataset", type=str, help="The name of the dataset to load", required=True
    )
    parser.add_argument(
        "--use-cache",
        default=False,
        action="store",
        help="If s3://<bucket>/<dataset>/parsed exists, its contents will be used",
    )
    parser.add_argument(
        "--cutover",
        default=False,
        action="store",
        help="If true, start routing requests to the Neo4j service",
    )
    parser.add_argument(
        "--remove-existing",
        default=False,
        action="store",
        help="If true, delete existing data in this dataset",
    )
    parser.add_argument(
        "--remap-ids", default=False, action="store", help="If true, remap UUIDs"
    )

    parser.add_argument(
        "--statistics",
        default=False,
        action="store",
        help="Produce import statistics, writing the output to `statistics.json`",
    )
    parser.add_argument(
        "--smoke-test", default=True, action="store", help="Run a post-load smoke-test"
    )

    args = parser.parse_args()

    dataset = args.dataset.strip()
    bucket = args.bucket.strip()

    if not bucket:
        print("missing bucket")
        parser.print_help()
        sys.exit(1)

    if not dataset:
        print("missing dataset")
        parser.print_help()
        sys.exit(1)

    load(
        dataset=dataset,
        bucket=bucket,
        use_cache=parse_truthy(args.use_cache),
        cutover=parse_truthy(args.cutover),
        remove_existing=parse_truthy(args.remove_existing),
        statistics=parse_truthy(args.statistics),
        smoke_test=parse_truthy(args.smoke_test),
        remap_ids=parse_truthy(args.remap_ids),
    )
