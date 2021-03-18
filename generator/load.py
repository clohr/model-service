import argparse
import json
import uuid
from csv import DictReader
from itertools import zip_longest
from os.path import dirname, isabs, join, realpath
from typing import Optional

from server.db import Config, Database, PartitionedDatabase
from server.models import (
    DatasetId,
    DatasetNodeId,
    ModelProperty,
    OrganizationId,
    OrganizationNodeId,
    UserNodeId,
)
from server.models import datatypes as dt

LOAD_CHUNK_SIZE: int = 2500


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def to_property_values(json_str):
    return {r["name"]: r["value"] for r in json.loads(json_str)}


def resolve_file(file_path, base_dir):
    if isabs(file_path):
        return file_path
    if base_dir is None:
        return realpath(file_path)
    return realpath(join(base_dir, file_path))


def load(db, input_file: str, verbose: bool = False, base_dir: Optional[str] = None):
    with open(input_file) as f:

        seed_files = json.load(f)

        # "models", "properties", and "records"
        for seed_file in seed_files:

            # Models
            with open(resolve_file(seed_file["model"], base_dir), "r") as model_file:
                models = DictReader(model_file, delimiter="|")
                model = next(models)
                model.pop("id")
                if verbose:
                    print(
                        f"{db.organization_id}:{db.dataset_id} :: Loading model {model['name']}"
                    )
                    print(model)
                model_id = db.create_model(**model).id
                if verbose:
                    print(
                        f"{db.organization_id}:{db.dataset_id} :: Created model {model_id}"
                    )

            # Model properties
            with open(
                resolve_file(seed_file["properties"], base_dir), "r"
            ) as properties_file:
                properties = list(DictReader(properties_file, delimiter="|"))
                properties[0]["model_title"] = True

                if verbose:
                    print(
                        f"{db.organization_id}:{db.dataset_id} :: Loading {len(properties)} properties"
                    )
                for prop in properties:
                    data_type = dt.deserialize(prop.pop("data_type"))
                    db.update_properties(
                        model_id, ModelProperty(data_type=data_type, **prop)
                    )

            # Records
            with open(
                resolve_file(seed_file["records"], base_dir), "r"
            ) as records_file:
                records_reader = DictReader(records_file, delimiter="|")
                total_loaded = 0
                for chunk in grouper(LOAD_CHUNK_SIZE, records_reader):
                    chunk = [r for r in chunk if r]
                    record_chunk = [to_property_values(row["values"]) for row in chunk]
                    db.create_records(model_id, record_chunk)
                    total_loaded += len(record_chunk)
                    if verbose:
                        print(
                            f"{db.organization_id}:{db.dataset_id} :: {len(record_chunk)} record(s), total {total_loaded}"
                        )
                if verbose:
                    print(
                        f"{db.organization_id}:{db.dataset_id} :: total records = {total_loaded}"
                    )
            if verbose:
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load seed data into Neo4j")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default=realpath(join(dirname(__file__), "output", "seed.json")),
    )
    parser.add_argument("--organization-id", type=int, default=1)
    parser.add_argument("--organization-node-id", type=str, default=str(uuid.uuid4()))
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-node-id", type=str, default=str(uuid.uuid4()))
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--user-node-id", type=str, default=str(uuid.uuid4()))
    parser.add_argument("--records", "-n", dest="n", type=int, default=1000)

    args = parser.parse_args()

    raw_db = Database.from_config(Config())

    with raw_db.transaction() as tx:
        raw_db.initialize_organization_and_dataset(
            tx,
            organization_id=OrganizationId(args.organization_id),
            dataset_id=DatasetId(args.dataset_id),
            organization_node_id=OrganizationNodeId(args.organization_node_id),
            dataset_node_id=DatasetNodeId(args.dataset_node_id),
        )

    db = PartitionedDatabase(
        raw_db,
        OrganizationId(args.organization_id),
        DatasetId(args.dataset_id),
        UserNodeId(args.user_node_id),
        OrganizationNodeId(args.organization_node_id),
        DatasetNodeId(args.dataset_node_id),
    )

    load(db, args.input, verbose=True)

    db.db.driver.close()

    print("done")
