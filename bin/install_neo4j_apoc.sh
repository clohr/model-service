#!/usr/bin/env bash

TARGET_DIR=$1
NEO4J_APOC_VERSION=$2
DOWNLOAD_URL="https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/$NEO4J_APOC_VERSION/apoc-$NEO4J_APOC_VERSION-all.jar"
DEST_DIR=$(realpath "$TARGET_DIR/apoc-$NEO4J_APOC_VERSION-all.jar")

if [ ! -e "$DEST_DIR" ]; then
  wget -O "$DEST_DIR" "$DOWNLOAD_URL"
fi

