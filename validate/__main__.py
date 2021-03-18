import argparse
import logging

import structlog

from core.config import Config
from server.db import Database, constraints
from utils.ssh import SSHTunnel
from utils.ssm import SSMParameters

if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=structlog.dev.ConsoleRenderer())
    )
    log = logging.getLogger()
    log.handlers = [handler]
    log.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Neptune to Neo4j migration script")

    parser.add_argument(
        "--environment",
        default="dev",
        help="The Pennsieve environment to validate (dev or prod or prd)",
    )

    parser.add_argument(
        "--jumpbox",
        default=None,
        help="The SSH jumpbox alias to route connections through",
    )

    args = parser.parse_args()

    if args.environment not in ["dev", "prod", "prd"]:
        raise Exception(f"Invalid environment {args.environment}")

    ssm_parameters = SSMParameters(args.environment)

    logging.getLogger().info(
        f"Validating structure of {args.environment} Neo4j database {ssm_parameters.neo4j_host}. Connecting via jumpbox: {args.jumpbox}"
    )

    with SSHTunnel(
        remote_host=ssm_parameters.neo4j_host,
        remote_port=ssm_parameters.neo4j_port,
        local_port=8888,
        jumpbox=args.jumpbox,
    ) as neo4j_tunnel:

        db = Database(
            uri=f"bolt://{neo4j_tunnel.host}:{neo4j_tunnel.port}",
            user=ssm_parameters.neo4j_user,
            password=ssm_parameters.neo4j_password,
            max_connection_lifetime=300,
        )

        constraints.check_integrity(db)

    log.info("Done. All OK.")
