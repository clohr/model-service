#!/usr/bin/env python

import argparse

import waitress  # type: ignore

from server.app import create_app  # type: ignore
from server.config import Config
from server.db import Database, index
from server.logging import configure_logging

parser = argparse.ArgumentParser(description="Pennsieve Model Service v2")

parser.add_argument(
    "-H", "--host", type=str, required=False, default="0.0.0.0", help="Server host"
)
parser.add_argument(
    "-P", "--port", type=int, required=False, default=8080, help="Server port"
)
parser.add_argument(
    "-T", "--threads", type=int, required=False, default=4, help="Number of threads"
)

if __name__ == "__main__":
    args = parser.parse_args()

    log = configure_logging()

    config = Config()
    db = Database.from_config(config)

    app = create_app(db=db, config=config)
    waitress.serve(app, host=args.host, port=args.port, threads=args.threads)
