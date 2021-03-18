#!/usr/bin/env bash

BASE=$(dirname "${BASH_SOURCE[0]}")
CONTEXT=$(realpath "$BASE/..")
DATA_DIR="$CONTEXT/data"

IMAGE_TAG=$1

if [ -z "$IMAGE_TAG" ]; then
  echo "Usage: $0 <image-tag>"
  exit 1
fi

if [ -z "$DATA_DIR" ]; then
  echo "$DATA_DIR directory is missing"
  exit 1
fi

TEMPDIR=$(mktemp -d)
DATA_DEST="$TEMPDIR/data-snapshot"

echo "Copy $DATA_DIR to $DATA_DEST"
cp -r "$(basename $DATA_DIR)" $DATA_DEST

chmod -R 777 "$DATA_DEST"

docker build -f Dockerfile.snapshot -t "pennsieve/neo4j-ppmi:$IMAGE_TAG" $TEMPDIR

