import argparse
import logging

import structlog

from .core import (
    delete_all_orphaned_datasets,
    delete_orphaned_datasets,
    generate_data,
    migrate_dataset,
)

if __name__ == "__main__":

    # Configure logging
    # ===================================================================

    shared_processors = [
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Prepare structlog events for the stdlib formatter
    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Aggregate stdlib and structlog and render to JSON
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(), foreign_pre_chain=shared_processors
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)

    # Parse arguments
    # ===================================================================

    parser = argparse.ArgumentParser(description="Neptune to Neo4j migration script")

    parser.add_argument(
        "--organization", type=str, help="The organization ID", required=True
    )

    parser.add_argument(
        "--dataset",
        help="The dataset ID, or 'all' to migrate all datasets in the organization",
        required=False,
    )

    parser.add_argument(
        "--environment",
        required=True,
        help="The Pennsieve environment to perform the migration in (dev or prod)",
    )

    parser.add_argument(
        "--jumpbox",
        default=None,
        help="The SSH jumpbox alias to route connections through",
    )

    parser.add_argument(
        "--remove-existing",
        default=False,
        action="store_true",
        help="If true, delete existing data in this dataset",
    )

    parser.add_argument(
        "--remap-ids",
        default=False,
        action="store_true",
        help="If true, remap UUIDs for all entities in the graph",
    )

    parser.add_argument(
        "--no-smoke-test",
        default=False,
        action="store_true",
        help="If set, don't run a smoke test",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force action",
        required=False,
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Load data into a dataset",
        required=False,
    )

    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Enable migrate mode",
        required=False,
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Enable dataset delete mode",
        required=False,
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file",
        required=False,
    )

    args = parser.parse_args()

    if args.environment not in ["dev", "prd"]:
        raise Exception(f"Invalid environment: {args.environment}")

    if args.jumpbox is None:
        logging.getLogger().warn(f"No jumpbox specified")

    if args.delete:
        if args.organization == "all":
            delete_all_orphaned_datasets(
                environment=args.environment,
                jumpbox=args.jumpbox,
                dry_run=not args.force,
            )
        else:
            organization_id = int(args.organization)
            delete_orphaned_datasets(
                organization_id=organization_id,
                environment=args.environment,
                jumpbox=args.jumpbox,
                dry_run=not args.force,
            )
    elif args.generate:
        generate_data(
            organization_id=args.organization,
            input_file=args.input_file,
            environment=args.environment,
            jumpbox=args.jumpbox,
            dry_run=not args.force,
        )
    elif args.migrate:
        if args.dataset == "all":
            dataset_id = None
        else:
            dataset_id = [int(args.dataset)]
        migrate_dataset(
            organization_id=args.organization,
            dataset_ids=dataset_id,
            remove_existing=args.remove_existing,
            environment=args.environment,
            jumpbox=args.jumpbox,
            smoke_test=not args.no_smoke_test,
            remap_ids=args.remap_ids,
        )
    else:
        print("Must provide --migrate or --delete")
