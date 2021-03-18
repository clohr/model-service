#!/usr/bin/env bash

BASE=$(dirname $(realpath $0))

source "$BASE/../test.env"

PENNSIEVE_API_HOST=$PENNSIEVE_API_HOST \
NEO4J_BOLT_URL=bolt://localhost:7687 \
NEO4J_BOLT_USER=$NEO4J_BOLT_USER \
NEO4J_BOLT_PASSWORD=$NEO4J_BOLT_PASSWORD \
NEO4J_AUTH=$NEO4J_AUTH \
S3_DATA_BUCKET=$S3_DATA_BUCKET python "$BASE/import_to_neo4j.py" $@
