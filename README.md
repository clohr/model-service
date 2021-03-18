# model-service

This project provides the model-service that interacts with the NEO4J database.

## Installing dependencies

To install all the required packages, run:

    $ make install

## Testing

Start a local Neo4j instance:

    $ docker-compose up -d neo4j

Then run the tests:

    $ make test

To skip integration tests that connect to the Pennsieve platform using the
Pennsieve Python client:

    $ pytest --skip-integration

## Types and Formatting

The CI build runs `mypy`, `black` and `isort` to typecheck the project and
detect any style issues.  Before submitting a PR, run the following to fix any
problems with code formatting:

    $ make format

To check that the `mypy` types line up, run

    $ make typecheck

Note: `make test` runs both `make format` and `make typecheck` under the hood.

## Exporting Data

From `pennsieve-api`:

    $ sh neptune-export/bin/exportDataset.sh -e dev -j non-prod -o $ORGANIZATION_INT_ID -d $DATASET_INT_ID

## Importing data

### From Neptune

    $ sh import-dataset.sh -e dev -j non-prod -d $ORGANIZATION_INT_ID/$DATASET_INT_ID

For import: If you pass `--cutover` to the import script, it will set the
`:Dataset.migrated_to_neo4j` flag to `TRUE`.

### Conventions

The `model-service` importer ingests CSV files stored on AWS S3 in the bucket

    dev-neo4j-pennsieve

Datasets are stored according to the convention

    dev-neo4j-pennsieve/
     <dataset>/
       raw/
         somefile.csv
         another_file.csv
         ...
         more-data.csv
       parsed/
         <processed files go here>

#### Example: PPMI

    dev-neo4j-pennsieve/
     <dataset>/
       raw/
         activity_assessment.csv
         adverse_event.csv
         ...
         whole_blood_collection.csv
       parsed/
         <processed files go here>

## Logging

Logging levels can be toggled via the `LOG_LEVEL` environment variable. The
levels `DEBUG`, `INFO`, `WARN`, and, `ERROR` are supported.

## Benchmarking

To run the Artillery load/performance benchmarks:

    cd load-test
    npm install
    ./artillery run load.yml

To debug an Artillery script:

    DEBUG='http:*' ./artillery run load.yml

