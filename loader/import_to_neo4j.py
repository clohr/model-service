import copy
import csv
import io
import itertools
import json
import logging
import os
import os.path
import re
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import boto3
from dateutil import parser as iso8601  # type: ignore

from server.auth import AuthContext
from server.config import Config
from server.db import Database, PartitionedDatabase, index, labels
from server.models import DatasetId, OrganizationId, RecordId
from server.models import datatypes as dt
from server.models import normalize_relationship_type


def connect(config: Config):
    return Database.from_config(config).driver


driver = None
s3_client = None
auth_context = None

LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", None)
DRY_RUN_ENABLED = False
PRODUCE_STATISTICS = False

BATCH_SIZE = 100_000
# BATCH_SIZE = 2


def parse_truthy(maybe):
    if isinstance(maybe, str):
        return maybe.lower().strip() in ("1", "true", "yes", "y", "t")
    try:
        return bool(maybe)
    except:  # noqa: E722
        return False


@dataclass(frozen=True)
class ModelRelationshipDefinition:
    id: str
    name: str
    from_model: Optional[str]
    to_model: Optional[str]
    display_name: str
    description: str
    created_at: str
    updated_at: str
    created_by: str
    updated_by: str

    def __str__(self):
        return f"({self.from_model or '*'})-[{self.name}]->({self.to_model or '*'})"

    @property
    def is_stub(self):
        assert not ((self.from_model is None) ^ (self.to_model is None))
        return self.from_model is None and self.to_model is None

    @property
    def relationship_type(self):
        return normalize_relationship_type(self.name)


@contextmanager
def session(driver):
    OUTPUT_LIMIT = 1024
    if DRY_RUN_ENABLED:

        def run(cmd, **kwargs):
            try:
                cmd = cmd.format(**kwargs)
                if len(cmd) > OUTPUT_LIMIT:
                    print("CYPHER $ " + cmd[:OUTPUT_LIMIT] + "...")
                else:
                    print("CYPHER $ " + cmd)
            except:  # noqa: E722
                print("CYPHER $ " + cmd)
            print("-" * 8)

        yield run
    else:
        with driver.session() as session:

            def run(cmd, **kwargs):
                logging.debug(cmd)
                logging.debug("-" * 8)
                return session.run(cmd, **kwargs).records()

            yield run


def dataset_files(s3_bucket, dataset, filetype=None):
    """
    Creates a generator which yields a (bucket, key) pair every iteration.

    Parameters
    ----------
    s3_bucket : str

    dataset : str

    filetype : str?

    Returns
    -------
    function
        a generator which yields a (bucket, dataset, key) pair every iteration
    """
    prefix = dataset
    if filetype is not None:
        prefix += "/" + filetype
    response = s3_client.list_objects(Bucket=s3_bucket, Prefix=prefix)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".csv"):
            yield (s3_bucket, dataset, key)


def raw_files(bucket, dataset):
    return dataset_files(bucket, dataset, filetype="raw")


def remapped_files(bucket, dataset):
    return dataset_files(bucket, dataset, filetype="remapped")


def parsed_files(bucket, dataset):
    return dataset_files(bucket, dataset, filetype="parsed")


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def as_filename(key):
    """
    Get the "<filename>.<ext>" portion of an S3 key

    Returns
    -------
    str
    """
    return os.path.basename(key)


def raw_key(dataset, filename):
    return "{}/raw/{}".format(dataset, filename)


def remapped_key(dataset, filename):
    return "{}/remapped/{}".format(dataset, filename)


def parsed_key(dataset, filename):
    return "{}/parsed/{}".format(dataset, filename)


def test_key(dataset, filename):
    return "{}/test/{}".format(dataset, filename)


def presign_file(bucket, dataset, filename, expiration_secs=3600):
    """
    Presign a file on S3.

    Parameters
    ----------
    bucket : str

    dataset : str

    filename : str

    expiration_secs : int
        The URL expiration time, in seconds. By default 3600 (1 hour)

    Returns
    -------
    str
        the presigned url
    """
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": parsed_key(dataset, filename)},
        ExpiresIn=expiration_secs,
    )


def fetch_file(bucket, dataset, key, type="parsed"):
    """
    Fetch a file from S3

    Returns
    -------
    file
        The fetched file
    """
    data_file = NamedTemporaryFile(mode="w+b")
    if type is None or type == "parsed":
        key = parsed_key(dataset, key)
    elif type == "test":
        key = test_key(dataset, key)

    if LOCAL_DATA_DIR:
        path = os.path.join(LOCAL_DATA_DIR, key)
        logging.info("Reading {} locally".format(path))
        return open(os.path.join(LOCAL_DATA_DIR, key), "rb")

    size = float(file_size(bucket, key))
    total_recv = 0.0

    def cb(bytes_recv):
        nonlocal total_recv
        bytes_recv = float(bytes_recv)
        percent = (total_recv / size) * 100.0 if size > 0.0 else 0.0
        logging.debug("fetch_file: {}/{} -- {:3.2f}%".format(bucket, key, percent))
        total_recv += bytes_recv

    logging.info("Downloading file from s3://{}/{}".format(bucket, key))
    s3_client.download_fileobj(bucket, key, data_file, Callback=cb)
    # Reset to the beginning:
    data_file.seek(0)
    return data_file


def file_size(bucket, key):
    """
    Get the size of a file on S3.

    Returns
    -------
    int
        The file size
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        logging.error("file_size failed: {}/{}".format(bucket, key))
        raise e
    return response["ContentLength"]


def assert_file_does_not_exist(bucket, key):
    """
    Check that a given file does not exist in S3.

    Raises an exception if it does exist.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        if e.response["Error"]["Code"] == "404":
            pass
        else:
            raise
    else:
        raise Exception(f"Did not expect to find file {key}")


def fetch_json(bucket, dataset, key, type="parsed"):
    """
    Fetch a JSON file from S3, parsing it as JSON and returning its contents

    Returns
    -------
    any
        The parsed contents
    """
    return json.load(fetch_file(bucket, dataset, key, type=type))


def store_file(bucket, dataset, key, fileobj):
    """
    Store a file on S3

    Parameters
    ----------
    bucket : str

    dataset : str

    key : str

    fileobj : fileobj
    """
    key = parsed_key(dataset, key)
    total_sent = 0

    logging.info("store_file: {}/{}".format(bucket, key))

    def cb(bytes_sent):
        nonlocal total_sent
        total_sent += bytes_sent
        logging.debug(
            "store_file: {}/{} -- transferred {} bytes".format(bucket, key, total_sent)
        )

    s3_client.upload_fileobj(fileobj, bucket, key, Callback=cb)


def store_json(bucket, dataset, key, to_json):
    """
    Store data on S3

    Parameters
    ----------
    bucket : str

    dataset : str

    key : str

    to_json : any

    Returns
    -------
    (str, str)
        the S3 (bucket, key) location of the stored JSON file
    """
    json_file = NamedTemporaryFile(mode="w+b")
    json_file.write(json.dumps(to_json).encode())
    # Reset to the beginning:
    json_file.seek(0)
    store_file(bucket, dataset, key, json_file)
    return (bucket, key)


def parse_header(header):
    parsed_header = list()
    for item in header:

        # Parse header name created by Neptune
        match = re.search(r"^\s*([~$]?[^:]*)[:]*([\w\[\]]*)$", item)
        if match:
            (name, vtype) = match.groups()

            parsed_header.append({"name": name, "vtype": vtype})
        else:
            raise Exception("Could not match header: {}".format(item))

    return parsed_header


def path_in_dataset(dataset, key):
    stripped = key
    if key.startswith(dataset + "/"):
        stripped = stripped[len(dataset + "/") :]

    if stripped.startswith("parsed/"):
        stripped = stripped[len("parsed/") :]
    elif stripped.startswith("raw/"):
        stripped = stripped[len("raw/") :]
    elif stripped.startswith("remapped/"):
        stripped = stripped[len("remapped/") :]

    return stripped


def create_records(bucket, dataset, headers):
    """
    Parameters
    ----------

    bucket : str

    dataset : str

    headers : dict
    """
    # Iterate over all model files and create records for all models
    record_counts = Counter()  # model-name -> count

    for (_, _, key) in dataset_files(bucket, dataset, filetype="parsed/records"):

        filename = as_filename(key)

        logging.info(
            "create_records: processing {} (filename={})".format(key, filename)
        )

        cur_headers = headers[path_in_dataset(dataset, key)]["props"]
        model = headers[path_in_dataset(dataset, key)]["model"]

        prop_str = ""
        for item in cur_headers:

            # These are always present and are set in the Cypher query below
            if item["name"] in [
                "~id",
                "~label",
                "$deleted",
                "$createdBy",
                "$updatedBy",
                "$createdAt",
                "$updatedAt",
                "$datasetId",
                "$organizationId",
            ]:
                continue

            elif item["vtype"] == "Double":
                prop_str += " `{name}`: toFloat(apoc.json.path(data.`{name}`)),".format(
                    **item
                )
            elif item["vtype"] == "Double[]":
                prop_str += " `{name}`: [x in apoc.json.path(data.`{name}`) | toFloat(x)],".format(
                    **item
                )
            elif item["vtype"] == "Long":
                prop_str += " `{name}`: toInt(apoc.json.path(data.`{name}`)),".format(
                    **item
                )
            elif item["vtype"] == "Long[]":
                prop_str += " `{name}`: [x in apoc.json.path(data.`{name}`) | toInt(x)],".format(
                    **item
                )
            elif item["vtype"] == "Date":
                prop_str += (
                    " `{name}`: datetime(apoc.json.path(data.`{name}`)),".format(**item)
                )
            elif item["vtype"] == "Date[]":
                prop_str += " `{name}`: [x in apoc.json.path(data.`{name}`) | datetime(x)],".format(
                    **item
                )
            elif item["vtype"] == "Boolean":
                prop_str += (
                    " `{name}`: toBoolean(apoc.json.path(data.`{name}`)),".format(
                        **item
                    )
                )
            elif item["vtype"] == "Boolean[]":
                prop_str += " `{name}`: [x in apoc.json.path(data.`{name}`) | toBoolean(x)],".format(
                    **item
                )
            elif item["vtype"] == "String":
                prop_str += (
                    " `{name}`: toString(apoc.json.path(data.`{name}`)),".format(**item)
                )
            elif item["vtype"] == "String[]":
                prop_str += " `{name}`: [x in apoc.json.path(data.`{name}`) | toString(x)],".format(
                    **item
                )
            else:
                raise Exception(
                    f"Do not know how to import property with header {item}"
                )

        prop_str = prop_str[:-1]

        done = False
        for i in itertools.count():
            if done:
                break

            logging.info(f"Importing batch {i}")

            presigned_url = presign_file(bucket, dataset, "records/" + filename)

            cmd = f"""
            CALL apoc.periodic.iterate(

            "LOAD CSV WITH HEADERS FROM $presigned_url AS data RETURN data SKIP $skip LIMIT $limit ",

            "MATCH ({labels.model("m")} {{ name: $model }})
                  -[{labels.in_dataset()}]->({{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({{ id: $organization_id }})

            CREATE ({labels.record("r")} {{
              `@id`: data.`~id`,
              `@sort_key`: 0
              {"," if prop_str else ""}
              {prop_str}
            }})

            CREATE (r)-[{labels.instance_of()}]->(m)

            SET m.`@max_sort_key` = m.`@max_sort_key` + 1
            SET r.`@sort_key` = m.`@max_sort_key`

            MERGE ({labels.user("createdBy")} {{ node_id: data.`$createdBy` }})
            MERGE ({labels.user("updatedBy")} {{ node_id: data.`$updatedBy` }})
            CREATE (r)-[{labels.created_by()} {{ at: datetime(data.`$createdAt`) }}]->(createdBy)
            CREATE (r)-[{labels.updated_by()} {{ at: datetime(data.`$updatedAt`) }}]->(updatedBy)

            RETURN r",

            {{
               batchSize: 1000,
               iterateList: true,
               retries: 3,
               params: {{
                 presigned_url: $presigned_url,
                 dataset_id: $dataset_id,
                 organization_id: $organization_id,
                 model: $model,
                 skip: $skip,
                 limit: $limit
              }}
            }})
            """
            with session(driver) as run:
                nodes = run(
                    cmd,
                    presigned_url=presigned_url,
                    model=model,
                    dataset_id=auth_context.dataset_id,
                    organization_id=auth_context.organization_id,
                    skip=i * BATCH_SIZE,
                    limit=BATCH_SIZE,
                )
                for node in nodes:
                    record_counts[model] += node["committedOperations"]

                    if node["failedBatches"] > 0:
                        logging.error(node)
                        raise Exception(node)

                    if node["failedOperations"] > 0:
                        logging.error(node)
                        raise Exception(node)

                    if node["committedOperations"] < BATCH_SIZE:
                        done = True

    return {
        "create_records": {
            "records": record_counts,
            "count": sum(record_counts.values()),
        }
    }


def create_models(bucket, dataset):
    """
    create_model_entities

    Parameters
    ----------
    bucker : str

    dataset : str

    headers : dict
    """
    with session(driver) as run:
        logging.info("Creating model nodes")

        presigned_url = presign_file(bucket, dataset, "models.csv")
        cmd = f"""
        USING PERIODIC COMMIT 500 LOAD CSV WITH HEADERS FROM $presigned_url AS data

        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        CREATE ({labels.model("m")} {{
          `@max_sort_key`: 0,
          id: data.`~id`,
          name: data.`$name`,
          display_name: data.`$displayName`,
          description: COALESCE(data.`$description`, ""),
          template_id: CASE data.`$templateId` WHEN "" THEN NULL ELSE data.`$templateId` END
        }})-[{labels.in_dataset()}]->(d)

        MERGE ({labels.user("createdBy")} {{ node_id: data.`$createdBy` }})
        MERGE ({labels.user("updatedBy")} {{ node_id: data.`$updatedBy` }})
        CREATE (m)-[{labels.created_by()} {{ at: datetime(data.`$createdAt`) }}]->(createdBy)
        CREATE (m)-[{labels.updated_by()} {{ at: datetime(data.`$updatedAt`) }}]->(updatedBy)
        RETURN m
        """
        nodes = run(
            cmd,
            organization_id=auth_context.organization_id,
            dataset_id=auth_context.dataset_id,
            presigned_url=presigned_url,
        )
        models = [node["m"] for node in nodes]
        return {
            "create_models": {
                "models": {m["id"]: m["name"] for m in models},
                "count": len(models),
            }
        }


def create_model_properties(bucket, dataset):
    """
    Load the model properties for the dataset.
    """
    with fetch_file(bucket, dataset, "propertyEdges.csv") as property_edges_file:
        property_edge_reader = csv.DictReader(
            io.TextIOWrapper(property_edges_file, errors="strict")
        )
        # Mapping of property id to model id
        property_edges = {edge["~to"]: edge["~from"] for edge in property_edge_reader}

    with fetch_file(bucket, dataset, "properties.csv") as model_property_file:
        model_property_reader = csv.DictReader(
            io.TextIOWrapper(model_property_file, errors="strict")
        )
        # Model properties need to be linked to their model in one Cypher query,
        # so we need the id of the model in scope with the property for UNWIND:
        model_properties = []
        for model_property in model_property_reader:
            model_id = property_edges[model_property["~id"]]
            model_property["$model_id"] = model_id

            default_value = json.loads(model_property["$defaultValue"])
            if default_value == {}:
                model_property["$defaultValue"] = None
            else:
                model_property["$defaultValue"] = default_value

            # Round-trip the datatype to convert from `"String`"` to `{"type": "String", "format": None}`
            model_property["$dataType"] = dt.serialize(
                dt.deserialize(model_property["$dataType"])
            )
            model_properties.append(model_property)

    with session(driver) as run:
        q = f"""
        UNWIND $props AS data
        MATCH ({labels.model("m")} {{ id: data.`$model_id` }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        MATCH (m)-[{labels.created_by()}]->({labels.user("createdBy")})
        MATCH (m)-[{labels.updated_by()}]->({labels.user("updatedBy")})

        CREATE ({labels.model_property("mp")} {{
          id:            data.`~id`,
          name:          toString(data.`$name`),
          display_name:  toString(data.`$displayName`),
          data_type:     data.`$dataType`, // ???
          description:   toString(data.`$description`),
          index:         toInteger(data.`$index`),
          locked:        toBoolean(data.`$locked`),
          default:       toBoolean(data.`$default`),
          default_value: data.`$defaultValue`,
          model_title:   toBoolean(data.`$conceptTitle`),
          required:      toBoolean(data.`$required`)
        }})
        CREATE (m)-[{labels.has_property()}]->(mp)

        // concepts-service does not store createdBy/updatedBy for properties.
        // However, it does store timestamps.
        // We populate the properties with the provenance of the model.

        CREATE (mp)-[{labels.created_by()} {{ at: datetime(data.`$createdAt`) }}]->(createdBy)
        CREATE (mp)-[{labels.updated_by()} {{ at: datetime(data.`$updatedAt`) }}]->(updatedBy)

        RETURN m.id AS model_id,
               mp
        """
        nodes = run(
            q,
            props=model_properties,
            organization_id=auth_context.organization_id,
            dataset_id=auth_context.dataset_id,
        )

        model_property_counts = Counter()
        created_properties = []
        for node in nodes:
            created_properties.append(
                {
                    "model": node["model_id"],
                    "property": {"id": node["mp"]["id"], "name": node["mp"]["name"]},
                }
            )
            model_property_counts[node["model_id"]] += 1

        assert len(created_properties) == len(model_properties)

        return {
            "create_model_properties": {
                "properties": created_properties,
                "count": len(model_property_counts),
            }
        }

    for model in db.get_models():
        with db.transaction():
            db.assert_.single_model_title(tx, model)
            db.assert_.unique_property_names(tx, model)
            db.assert_.unique_property_display_names(tx, model)


def fetch_relationships(bucket, dataset, relation_types):
    """
    Get a listing of relationship files on S3

    Parameters
    ----------
    bucket : str

    dataset : str

    relationship_file : fileobj

    relation_types : List[str]

    Returns
    -------
    {str: [(str, str)}
        A dict of (bucket, key) pairs, where the key is the relation type
        and the value is the location of the corresponding relationship files.
    """
    relationship_locations = dict()

    for rel_type in relation_types:
        rel_file_key = "relationships/{}.csv".format(rel_type)
        logging.info("Fetching relationship {}".format(rel_file_key))
        relationship_locations[rel_type] = (bucket, rel_file_key)

    return relationship_locations


def create_record_relationships(
    db: PartitionedDatabase,
    bucket,
    dataset,
    proxy_model_relationships: Dict[str, ModelRelationshipDefinition],
    canonical_relationships: List[ModelRelationshipDefinition],
    canonical_relationship_remapping: Dict[str, str],
):
    """
    Parameters
    ----------
    bucket : str
        The data bucket

    dataset : str
        The dataset name
    """
    proxy_relationship_types = [r.name for r in proxy_model_relationships.values()]
    proxy_relationship_ids = [r.id for r in proxy_model_relationships.values()]

    logging.info(f"Ignoring proxy relationships {proxy_relationship_types}")

    with fetch_file(bucket, dataset, "relation_types.csv") as relation_types_file:
        relation_types_reader = csv.DictReader(
            io.TextIOWrapper(relation_types_file, errors="strict")
        )
        relation_types = [line["$name"] for line in relation_types_reader]

    # de-dupe:
    relation_types = set(relation_types)

    relationship_files = fetch_relationships(bucket, dataset, relation_types)

    record_relationships = Counter()

    canonical_relationships = {
        r.id: {"from_model": r.from_model, "to_model": r.to_model}
        for r in canonical_relationships
    }

    for relation_type in relation_types:
        logging.info(f"Creating record relationships for {relation_type}")

        # Generate a presigned URL for the relationship file:
        (rel_bucket, rel_key) = relationship_files[relation_type]
        presigned_url = presign_file(rel_bucket, dataset, rel_key)

        normalized_relation_type = normalize_relationship_type(relation_type)

        # Check that the every relationship has a canonical model relationship
        # and that each relationship connects to the correct models.

        done = False
        for i in itertools.count():
            if done:
                break

            logging.info(f"Validating batch {i}")

            cmd = f"""
            LOAD CSV WITH HEADERS FROM $presigned_url AS edge
            WITH edge {{ .*, model_relationship_id : $canonical_relationship_remapping[edge.`$schemaRelationshipId`] }} AS edge
            SKIP $skip LIMIT $limit

            OPTIONAL MATCH ({labels.record("from_record")} {{ `@id`: edge.`~from` }})
                  -[{labels.instance_of()}]->({labels.model("from_model")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
            USING INDEX {labels.record("from_record")}(`@id`)

            OPTIONAL MATCH ({labels.record("to_record")} {{ `@id`: edge.`~to` }})
                  -[{labels.instance_of()}]->({labels.model("to_model")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
            USING INDEX {labels.record("to_record")}(`@id`)

            WITH edge, from_record, to_record, from_model, to_model,

              // ===================================================================
              // There are two ways the record relationship can be malformed:

              // 1) If the schema relationship of the edge no longer exists
              // However, it is ok if either the "from" or "to" record *also* no longer exists.
              (edge.model_relationship_id IS NULL AND NOT (from_record IS NULL OR to_record IS NULL)) AS unknown_schema_relationship,

              // 2) If the model relationship connects models other than the models
              // specified by this model relationship
              ($canonical_relationships[edge.model_relationship_id].from_model <> from_model.id
               OR $canonical_relationships[edge.model_relationship_id].to_model <> to_model.id) AS mismatched_schema_relationship

              // ===================================================================

            WHERE unknown_schema_relationship OR mismatched_schema_relationship

            RETURN edge.`$schemaRelationshipId` AS schema_relationship_id,
                   edge.`~id` AS id,
                   edge,
                   unknown_schema_relationship,
                   mismatched_schema_relationship,
                   from_record,
                   to_record,
                   from_model,
                   to_model
            """
            with session(driver) as run:
                bad_relationships = list(
                    run(
                        cmd,
                        presigned_url=presigned_url,
                        organization_id=auth_context.organization_id,
                        dataset_id=auth_context.dataset_id,
                        canonical_relationship_remapping=canonical_relationship_remapping,
                        canonical_relationships=canonical_relationships,
                        skip=i * BATCH_SIZE,
                        limit=BATCH_SIZE,
                    )
                )

            # Each broken record relationship is individually rewritten
            # (mapping record ID to a new model relationship ID)
            individual_relationship_remapping = {}

            model_cache = {}

            for node in bad_relationships:
                if node["unknown_schema_relationship"]:
                    logging.warning(
                        f"Could not rewrite schemaRelationshipId {node['schema_relationship_id']} for record relationship {node['id']}"
                    )

                if node["mismatched_schema_relationship"]:
                    target = canonical_relationships[
                        node["edge"]["model_relationship_id"]
                    ]
                    logging.warning(
                        f"Relationship {node['id']} connects records of model "
                        f"{node['from_model']['name']} ({node['from_model']['id']}) -[{relation_type}]-> {node['to_model']['name']} ({node['to_model']['id']}) "
                        f"but model relationship is defined between "
                        f"{target['from_model']} -[{relation_type}]-> {target['to_model']}"
                    )

                # These two issues have the same solution: find the correct model relationship
                # (same "from model", "name", and "to model") and use it instead.

                if (
                    node["unknown_schema_relationship"]
                    or node["mismatched_schema_relationship"]
                ):
                    # Find a replacement model relationship, if it exists in the graph
                    key = (
                        node["from_model"]["id"],
                        relation_type,
                        node["to_model"]["id"],
                    )

                    if key in model_cache:
                        model = model_cache[key]
                    else:
                        with db.transaction() as tx:
                            matches = db.get_model_relationships_tx(
                                tx,
                                node["from_model"]["id"],
                                relation_type,
                                node["to_model"]["id"],
                                one_to_many=True,
                            )
                            if len(matches) != 1:
                                raise Exception(
                                    "Could not find a compatible model relationship to rewrite to"
                                )
                            model = matches[0]
                            model_cache[key] = model

                    assert node["id"] not in individual_relationship_remapping
                    individual_relationship_remapping[node["id"]] = model.id

            # Use apoc.periodic.iterate to avoid "Eager" statement in the query plan.
            # Eager means that the entire query set will be held in memory, even when
            # using `PERIODIC COMMIT`
            # https://neo4j.com/docs/cypher-manual/current/execution-plans/operators/#query-plan-eager
            #
            # The only downside is that `iterate` does not return results, so the
            # preflight check must be done in a separate query. See above.

            logging.info(f"Creating batch {i}")

            presigned_url = presign_file(rel_bucket, dataset, rel_key)

            cmd = f"""
            CALL apoc.periodic.iterate(

            "LOAD CSV WITH HEADERS FROM $presigned_url AS edge RETURN edge SKIP $skip LIMIT $limit",

            "WITH edge,
                 datetime(edge.`$updatedAt`) AS updated_at,
                 datetime(edge.`$createdAt`) AS created_at,
                 edge.`$createdBy` AS created_by,
                 edge.`$updatedBy` AS updated_by,
                 COALESCE(
                   $individual_relationship_remapping[edge.`~id`],
                   $canonical_relationship_remapping[edge.`$schemaRelationshipId`]
                 ) AS model_relationship_id

            MATCH ({labels.record("a")} {{ `@id`: edge.`~from` }})
                  -[{labels.instance_of()}]->({labels.model("ma")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
            USING INDEX {labels.record("a")}(`@id`)

            MATCH ({labels.record("b")} {{ `@id`: edge.`~to` }})
                  -[{labels.instance_of()}]->({labels.model("mb")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
            USING INDEX {labels.record("b")}(`@id`)

            // This will ignore any relationships that connect records to proxy instances
            // because there are no record nodes with the ids of the proxy instances.

            MERGE (a)-[r:`{normalized_relation_type}`{{ model_relationship_id: model_relationship_id }}]->(b)
            ON CREATE SET
                r.id = edge.`~id`,
                r.created_at = created_at,
                r.updated_at = updated_at,
                r.created_by = created_by,
                r.updated_by = updated_by
            ON MATCH SET
                // Pick latest updated at and updated by
                r.updated_at = CASE updated_at > r.updated_at WHEN true THEN updated_at ELSE r.updated_at END,
                r.updated_by = CASE updated_at > r.updated_at WHEN true THEN updated_by ELSE r.updated_by END,

                // Pick earliest created at and created by
                r.created_at = CASE created_at < r.created_at WHEN true THEN created_at ELSE r.created_at END,
                r.created_by = CASE created_at < r.created_at WHEN true THEN created_by ELSE r.created_by END

            RETURN r.id                         AS id,
                   TYPE(r)                      AS type,
                   edge.`$schemaRelationshipId` AS schema_relationship_id,
                   r.model_relationship_id      AS model_relationship_id",

            {{
               batchSize: 1000,
               iterateList: true,
               retries: 3,
               params: {{
                 presigned_url: $presigned_url,
                 dataset_id: $dataset_id,
                 organization_id: $organization_id,
                 canonical_relationship_remapping: $canonical_relationship_remapping,
                 individual_relationship_remapping: $individual_relationship_remapping,
                 skip: $skip,
                 limit: $limit
              }}
            }})
            """
            with session(driver) as run:
                nodes = run(
                    cmd,
                    presigned_url=presigned_url,
                    organization_id=auth_context.organization_id,
                    dataset_id=auth_context.dataset_id,
                    canonical_relationship_remapping=canonical_relationship_remapping,
                    individual_relationship_remapping=individual_relationship_remapping,
                    skip=i * BATCH_SIZE,
                    limit=BATCH_SIZE,
                )

                for node in nodes:
                    record_relationships[relation_type] += node["committedOperations"]

                    if node["failedBatches"] > 0:
                        logging.error(node)
                        raise Exception(node)

                    if node["failedOperations"] > 0:
                        logging.error(node)
                        raise Exception(node)

                    if node["committedOperations"] < BATCH_SIZE:
                        done = True

    return {
        "create_record_relationships": {
            "relationships": record_relationships,
            "total": sum(record_relationships.values()),
        }
    }


def create_proxy_relationships(
    bucket, dataset, proxy_model_relationships: Dict[str, ModelRelationshipDefinition]
):

    proxy_relationship_types = [r.name for r in proxy_model_relationships.values()]

    # Create stub proxy nodes

    presigned_url = presign_file(bucket, dataset, "proxies.csv")

    proxy_relationship_counts = Counter()

    cmd = f"""
    USING PERIODIC COMMIT 500 LOAD CSV WITH HEADERS FROM $presigned_url AS data
    MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
       -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

    CREATE (tp:TempProxyConcept {{
      id: data.`~id`,
      external_id: data.`$externalId`,
      package_id: toInteger(data.`$packageId`),
      package_node_id: data.`$packageNodeId`,
      created_at: data.`$createdAt`,
      updated_at: data.`$updatedAt`,
      count: 0
    }})-[{labels.in_dataset()}]->(d)

    RETURN COUNT(tp) AS count
    """
    with driver.session() as s:
        count = (
            s.run(
                cmd,
                presigned_url=presigned_url,
                organization_id=auth_context.organization_id,
                dataset_id=auth_context.dataset_id,
            )
            .single()
            .get("count")
        )

        logging.info(f"Found {count} proxies")

    # de-dupe:
    proxy_relationship_types = set(proxy_relationship_types)

    proxy_relationship_files = fetch_relationships(
        bucket, dataset, proxy_relationship_types
    )

    logging.info(f"Found {len(proxy_relationship_files)} proxy relationship files")

    for relationship_type in proxy_relationship_types:
        # Generate a presigned URL for the relationship file:
        (rel_bucket, rel_key) = proxy_relationship_files[relationship_type]
        presigned_url = presign_file(rel_bucket, dataset, rel_key)

        # TODO: should we reverse these relationships????

        cmd = f"""
        USING PERIODIC COMMIT 500 LOAD CSV WITH HEADERS FROM $presigned_url AS data

        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
               -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        // This will fail to match any non-proxy relationships that were imported
        // from these CSVs in `create_record_relationships`

        // Proxy relationships can be created either to *or* from package proxies.
        // We need to handle both cases here:

        // (record)-[belongs_to]->(package)

        OPTIONAL MATCH (to_tp:TempProxyConcept {{ id: data.`~to` }})
                 -[{labels.in_dataset()}]->(d)

        OPTIONAL MATCH ({labels.record("from_r")} {{ `@id`: data.`~from` }})
                 -[{labels.instance_of()}]->({labels.model("m")})
                 -[{labels.in_dataset()}]->(d)
        USING INDEX {labels.record("from_r")}(`@id`)

        WITH d, o, to_tp, from_r, data

        // (package)-[belongs_to]->(record)

        OPTIONAL MATCH (from_tp:TempProxyConcept {{ id: data.`~from` }})
                 -[{labels.in_dataset()}]->(d)

        OPTIONAL MATCH ({labels.record("to_r")} {{ `@id`: data.`~to` }})
                 -[{labels.instance_of()}]->({labels.model("m")})
                 -[{labels.in_dataset()}]->(d)
        USING INDEX {labels.record("to_r")}(`@id`)

        WITH d, o, data, [
          {{ package: to_tp, record: from_r }},
          {{ package: from_tp, record: to_r }}
        ] AS possibilities
        UNWIND possibilities AS relationship
        WITH d, o, data, relationship.record AS record, relationship.package AS tp
        WHERE (NOT tp IS NULL) AND (NOT record IS NULL)

        MERGE ({labels.package("pp")} {{ package_id: tp.package_id }})
              -[{labels.in_dataset()}]->(d)
        ON CREATE SET pp.package_node_id = tp.package_node_id

        MERGE (pp)<-[{labels.in_package("proxy_relationship")} {{ relationship_type: $relationship_type }}]-(record)
        ON CREATE SET proxy_relationship.id                = data.`~id`,
                      proxy_relationship.proxy_instance_id = tp.id,
                      proxy_relationship.created_at        = data.`$createdAt`,
                      proxy_relationship.updated_at        = data.`$updatedAt`,
                      proxy_relationship.created_by        = data.`$createdBy`,
                      proxy_relationship.updated_by        = data.`$updatedBy`

        WITH pp, tp, proxy_relationship
        SET tp.count = tp.count + 1
        RETURN pp,
               proxy_relationship
        """
        with driver.session() as s:
            nodes = s.run(
                cmd,
                presigned_url=presigned_url,
                organization_id=auth_context.organization_id,
                dataset_id=auth_context.dataset_id,
                relationship_type=relationship_type,
            )
            for node in nodes:
                proxy_relationship_counts[node["pp"]["package_id"]] += 1

    cmd = f"""
    MATCH  ({labels.organization("o")} {{ id: $organization_id }})
          <-[{labels.in_organization()}]-({labels.dataset("d")} {{ id: $dataset_id }})
          <-[{labels.in_dataset()}]-(tpc:TempProxyConcept)
    WHERE tpc.count = 0
    RETURN COUNT(tpc)               AS orphaned_count,
           COLLECT(DISTINCT tpc.id) AS orphaned_ids
    """
    with driver.session() as s:
        result = s.run(
            cmd,
            organization_id=auth_context.organization_id,
            dataset_id=auth_context.dataset_id,
        ).single()
        orphaned_count = result.get("orphaned_count")
        orphaned_ids = result.get("orphaned_ids")

        if orphaned_count > 0:
            logging.warn(f"Found {orphaned_count} orphaned temp proxy nodes")
            logging.warn(f"==> [{', '.join(orphaned_ids)}]")

    cmd = f"""
    MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
          -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
    MATCH (tpc:TempProxyConcept)-[{labels.in_dataset()}]->(d)
    DETACH DELETE tpc
    RETURN COUNT(*) AS deleted_count
    """
    with driver.session() as s:
        deleted_count = (
            s.run(
                cmd,
                organization_id=auth_context.organization_id,
                dataset_id=auth_context.dataset_id,
                orphaned_ids=orphaned_ids,
            )
            .single()
            .get("deleted_count")
        )
        logging.info(f"Deleted {deleted_count} temp proxy nodes")

    return {
        "create_proxy_relationships": {
            "orphaned": {"ids": orphaned_ids, "count": orphaned_count},
            "proxy_relationships": {
                "to_packages": proxy_relationship_counts,
                "total": sum(proxy_relationship_counts.values()),
            },
        }
    }


def get_proxy_model_id(bucket, dataset) -> Optional[str]:
    """
    Return the ID of the proxy concept, if it exists.
    """
    with fetch_file(bucket, dataset, "proxy_model.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, errors="strict"))
        try:
            line = next(reader)
            return line["~id"]
        except StopIteration:
            return None


def create_model_relationships(bucket, dataset):
    """
    create_model_relationships

    Parameters
    ----------
    relationship_map : dict
    """
    models = {}
    with fetch_file(bucket, dataset, "models.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, errors="strict"))
        for line in reader:
            models[line["~id"]] = line["$name"]

    # Get all relationship types:
    with fetch_file(bucket, dataset, "relation_types.csv") as relation_types_file:
        lines = list(
            csv.DictReader(io.TextIOWrapper(relation_types_file, errors="strict"))
        )

        relationships = [
            ModelRelationshipDefinition(
                id=line["~id"],
                from_model=line["$from"].strip() or None,
                to_model=line["$to"].strip() or None,
                name=line["$name"],
                created_at=line["$createdAt"],
                updated_at=line["$updatedAt"],
                created_by=line["$createdBy"],
                updated_by=line["$updatedBy"],
                description=line["$description"] or "",
                display_name=line["$displayName"],
            )
            for line in lines
        ]

        relationship_types = {
            r.name: r for r in sorted(relationships, key=lambda r: r.is_stub)
        }

        # Deduplicate relationship stubs. This can happen with some older datasets.
        # ======================================================================
        relationship_stubs = [r for r in relationships if r.is_stub]

        if len(set(r.name for r in relationship_stubs)) < len(relationship_stubs):
            logging.warning(
                f"Deduplicating model relationship stubs by name: {[(r.name, r.id) for r in relationship_stubs]}"
            )

        relationship_stubs = {r.name: r for r in relationship_stubs}

    # And all schema relations
    with fetch_file(bucket, dataset, "schemaRelations.csv") as f:

        schema_relations = []

        # It is possible for a schema relationship to exist but not the corresponding
        # relationship type. This can happen because concepts-service does not
        # correctly delete schema relationships.  It deletes the parent
        # relationship type, but leaves the schema relationship behind.
        for line in csv.DictReader(io.TextIOWrapper(f, errors="strict")):
            relationship_name = line["$relationshipType"]

            if relationship_name not in relationship_types:
                logging.warning(
                    f"create_model_relationship: no relationship type for schema relationship {relationship_name}"
                )

                # Sanity check that this schema relationship does not have any relationship instances.
                key = parsed_key(dataset, f"relationships/{relationship_name}.csv")
                assert_file_does_not_exist(bucket, key)
                continue

            schema_relations.append(
                ModelRelationshipDefinition(
                    id=line["~id"],
                    from_model=line["~from"].strip() or None,
                    to_model=line["~to"].strip() or None,
                    name=relationship_name,
                    created_at=line["$createdAt"],
                    updated_at=line["$updatedAt"],
                    created_by=relationship_types[relationship_name].created_by,
                    updated_by=relationship_types[relationship_name].updated_by,
                    description=relationship_types[relationship_name].description,
                    display_name=relationship_types[relationship_name].display_name,
                )
            )

    # Create relationship stubs for all relationship types without "from" and "to"
    # ======================================================================
    model_relationship_stubs = defaultdict(list)

    for model_relationship in relationship_stubs.values():
        logging.info(f"creating model relationship stub: {model_relationship}")

        cmd = f"""
        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        CREATE ({labels.model_relationship_stub("r")} {{
                 id: $id,
                 name: $name,
                 display_name: $display_name,
                 description: $description,
                 type: $relationship_type,
                 created_at: $created_at,
                 updated_at: $updated_at,
                 created_by: $created_by,
                 updated_by: $updated_by
               }})
        CREATE (r)-[{labels.in_dataset()}]->(d)
        RETURN r
        """
        with session(driver) as run:
            nodes = list(
                run(
                    cmd,
                    dataset_id=auth_context.dataset_id,
                    organization_id=auth_context.organization_id,
                    relationship_type=model_relationship.relationship_type,
                    **asdict(model_relationship),
                )
            )
            assert len(nodes) == 1, "Did not create model relationship stub"

            for node in nodes:
                model_relationship_stubs[node["r"]["type"]].append(
                    {"id": node["r"]["id"]}
                )

    # Deduplicate model relationships
    # ======================================================================

    # We consider relationships to be duplicated if (from, to, name) are the same.
    duplicate_relationships = defaultdict(list)

    # Prefer relationships types for the canonical representation
    for model_relationship in relationships:
        if not model_relationship.is_stub:
            duplicate_relationships[
                (
                    model_relationship.from_model,
                    model_relationship.name,
                    model_relationship.to_model,
                )
            ].append(model_relationship)

    # In some cases there are schema relationships for a relationship stub.
    # We need to create model relationships for these as well
    for model_relationship in schema_relations:
        duplicate_relationships[
            (
                model_relationship.from_model,
                model_relationship.name,
                model_relationship.to_model,
            )
        ].append(model_relationship)

    # For every relationship that *is* duplicated, we pick a canonical
    # representation.  When creating record relationships we will need to
    # rewrite the schemaRelationshipId to this canonical schema relationship
    #
    # Prefer relationships from the `relationship_types` file, if they exist.
    canonical_relationships = []
    canonical_relationship_remapping = {}
    for relationships in duplicate_relationships.values():
        canonical_relationships.append(relationships[0])
        canonical_relationship_remapping.update(
            {r.id: relationships[0].id for r in relationships}
        )

    # Create model relationships
    # ======================================================================

    proxy_id = get_proxy_model_id(bucket, dataset)
    proxy_relationships = {}

    created_model_relationships = []
    orphaned_schema_relationships = []

    for model_relationship in canonical_relationships:

        # Proxy relationships should not be created here.  Accumulate the
        # proxy relationships, and use them later to build relationships to
        # the new :Package node.
        if proxy_id and (
            model_relationship.from_model == proxy_id
            or model_relationship.to_model == proxy_id
        ):
            logging.info(f"found proxy relationship {model_relationship.name}")
            proxy_relationships[model_relationship.id] = model_relationship
            continue

        elif model_relationship.to_model not in models:
            logging.warning(
                f"create_model_relationship: missing schema relationship :{model_relationship.name}: `to` target (model={model_relationship.to_model}) does not exist"
            )
            orphaned_schema_relationships.append(
                {
                    "id": model_relationship.id,
                    "type": model_relationship.relationship_type,
                    "from": model_relationship.from_model,
                    "to": model_relationship.to_model,
                    "missing": model_relationship.to_model,
                }
            )
            continue

        elif model_relationship.from_model not in models:
            logging.warning(
                f"create_model_relationship: missing schema relationship :{model_relationship.name}: `from` target (model={model_relationship.from_model}) does not exist"
            )
            orphaned_schema_relationships.append(
                {
                    "id": model_relationship.id,
                    "type": model_relationship.relationship_type,
                    "from": model_relationship.from_model,
                    "to": model_relationship.to_model,
                    "missing": model_relationship.from_model,
                }
            )
            continue

        logging.info(f"creating model relationship {model_relationship}")

        cmd = f"""
        MATCH  ({labels.model("m")} {{ id: $from_model }})
              -[{labels.in_dataset()}]->(d {{ id: $dataset_id }})
              -[{labels.in_organization()}]->(o {{ id: $organization_id }})
        MATCH  ({labels.model("n")} {{ id: $to_model }})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        CREATE (m)-[r:`{model_relationship.relationship_type}` {{
          one_to_many: true,
          id: $id,
          name: $name,
          display_name: $display_name,
          description: $description,
          created_at: datetime($created_at),
          updated_at: datetime($updated_at),
          created_by: $created_by,
          updated_by: $updated_by
        }}]->(n)

        CREATE (m)-[{labels.related_to("r_v2")} {{
          type: TYPE(r),
          one_to_many: true,
          id: $id,
          name: $name,
          display_name: $display_name,
          description: $description,
          created_at: datetime($created_at),
          updated_at: datetime($updated_at),
          created_by: $created_by,
          updated_by: $updated_by
        }}]->(n)

        RETURN r,
                   TYPE(r) AS type,
                   m.id    AS from_,
                   n.id    AS to
        """
        with session(driver) as run:
            nodes = list(
                run(
                    cmd,
                    dataset_id=auth_context.dataset_id,
                    organization_id=auth_context.organization_id,
                    **asdict(model_relationship),
                )
            )
            assert (
                len(nodes) == 1
            ), "Could not create model relationship {model_relationship}"

            for node in nodes:
                created_model_relationships.append(
                    {
                        "id": node["r"]["id"],
                        "type": node["type"],
                        "from": node["from_"],
                        "to": node["to"],
                    }
                )

    stats = {
        "create_model_relationships": {
            "success": {
                "relationships": created_model_relationships,
                "count": len(created_model_relationships),
            },
            "orphaned": {
                "relationships": orphaned_schema_relationships,
                "count": len(orphaned_schema_relationships),
            },
            "stubs": {
                "relationships": model_relationship_stubs,
                "count": len(model_relationship_stubs),
            },
        }
    }
    return (
        proxy_relationships,
        canonical_relationships,
        canonical_relationship_remapping,
        stats,
    )


def create_model_linked_properties(bucket, dataset):

    model_linked_properties = {}

    with fetch_file(bucket, dataset, "schemaLinkedProperties.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, errors="strict"))

        for item in reader:
            relationship_type = normalize_relationship_type(item["$name"])

            cmd = f"""
            MATCH  ({labels.model("m")} {{ id: $from_model }})
                  -[{labels.in_dataset()}]->(d {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->(o {{ id: $organization_id }})
            MATCH  ({labels.model("n")} {{ id: $to_model }})
                  -[{labels.in_dataset()}]->(d)
                  -[{labels.in_organization()}]->(o)

            CREATE (m)-[r:`{relationship_type}` {{
              one_to_many: false,
              id: $relationship_id,
              name: $name,
              display_name: $display_name,
              description: $description,
              created_at: datetime($created_at),
              updated_at: datetime($updated_at),
              created_by: $created_by,
              updated_by: $updated_by,
              index: toInteger($index)
            }}]->(n)

            CREATE (m)-[{labels.related_to()} {{
              type: TYPE(r),
              one_to_many: false,
              id: $relationship_id,
              name: $name,
              display_name: $display_name,
              description: $description,
              created_at: datetime($created_at),
              updated_at: datetime($updated_at),
              created_by: $created_by,
              updated_by: $updated_by,
              index: toInteger($index)
            }}]->(n)

            RETURN r,
                   m.id AS from,
                   n.id AS to
            """
            linked_property_relationship_type = []

            with session(driver) as run:
                nodes = list(
                    run(
                        cmd,
                        from_model=item["~from"],
                        to_model=item["~to"],
                        relationship_id=item["~id"],
                        created_at=item["$createdAt"],
                        updated_at=item["$updatedAt"],
                        created_by=item["$createdBy"],
                        updated_by=item["$updatedBy"],
                        dataset_id=auth_context.dataset_id,
                        organization_id=auth_context.organization_id,
                        name=item["$name"],
                        display_name=item["$displayName"],
                        description="",  # Not set for linked properties
                        index=item["$position"],
                    )
                )
                assert len(nodes) == 1, "Could not create model linked property {item}"

                for node in nodes:
                    linked_property_relationship_type.append(
                        {"id": node["r"]["id"], "from": node["from"], "to": node["to"]}
                    )

            model_linked_properties[relationship_type] = {
                "relationships": linked_property_relationship_type,
                "count": len(linked_property_relationship_type),
            }

    return {
        "create_model_linked_properties": {
            "linked_properties": model_linked_properties,
            "count": sum(mlp["count"] for mlp in model_linked_properties.values()),
        }
    }


def create_record_linked_properties(bucket, dataset, db):

    with fetch_file(bucket, dataset, "schemaLinkedProperties.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, errors="strict"))
        schema_linked_property_map = {item["~id"]: item for item in reader}

    with fetch_file(bucket, dataset, "linkedProperties.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, errors="strict"))

        # Deduplicate linked properties caused by https://app.clickup.com/t/y55gn
        # Linked property instances must be unique by (schema relationship, source record).
        linked_properties = defaultdict(list)
        for item in reader:

            # Check that the schema linked property still exists.
            #
            # In some older datasets, there are cases where the schema no longer
            # exists, *and* either the "from" or "to" record no longer exists.
            # This is bad data that can safely be ignored.
            if item["$schemaLinkedPropertyId"] not in schema_linked_property_map:
                if (
                    db.get_record(item["~from"]) is None
                    or db.get_record(item["~to"]) is None
                ):
                    continue
            else:
                assert schema_linked_property_map.get(item["$schemaLinkedPropertyId"])

            linked_properties[(item["$schemaLinkedPropertyId"], item["~from"])].append(
                item
            )
        unique_linked_properties = [
            max(
                duplicate_linked_properties,
                key=lambda item: iso8601.parse(item["$createdAt"]),
            )
            for duplicate_linked_properties in linked_properties.values()
        ]

        linked_property_counts = Counter()

        for item in unique_linked_properties:
            model_relationship = schema_linked_property_map[
                item["$schemaLinkedPropertyId"]
            ]

            relationship_type = normalize_relationship_type(model_relationship["$name"])

            cmd = f"""
            MATCH ({labels.record("a")} {{ `@id`: $from_record }})
                   -[{labels.instance_of()}]->({labels.model()} {{ id: $from_model_id }})
                   -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                   -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

            MATCH ({labels.record("b")} {{ `@id`: $to_record }})
                   -[{labels.instance_of()}]->({labels.model()} {{ id: $to_model_id }})
                   -[{labels.in_dataset()}]->(d)
                   -[{labels.in_organization()}]->(o)

            CREATE (a)-[r:`{relationship_type}` {{
              id: $id,
              model_relationship_id: $model_relationship_id,
              created_at: datetime($created_at),
              updated_at: datetime($updated_at),
              created_by: $created_by,
              updated_by: $updated_by
            }}]->(b)
            RETURN r
            """
            with session(driver) as run:
                nodes = list(
                    run(
                        cmd,
                        from_record=item["~from"],
                        to_record=item["~to"],
                        id=item["~id"],
                        model_relationship_id=item["$schemaLinkedPropertyId"],
                        created_at=item["$createdAt"],
                        updated_at=item["$updatedAt"],
                        created_by=item["$createdBy"],
                        updated_by=item["$updatedBy"],
                        from_model_id=model_relationship["~from"],
                        to_model_id=model_relationship["~to"],
                        dataset_id=auth_context.dataset_id,
                        organization_id=auth_context.organization_id,
                    )
                )
                assert len(nodes) == 1, f"Did not created linked property {item}"

                for node in nodes:
                    linked_property_counts[relationship_type] += 1

    return {
        "create_record_linked_properties": {
            "linked_properties": linked_property_counts,
            "count": sum(linked_property_counts.values()),
        }
    }


def extract_header(bucket, dataset, key, remap_ids=False):
    """
    Extracts the header of the given S3 asset, creating a translated instance
    of the asset with the new header, uploading the result to the S3 path
    `<bucket>/<dataset>/parsed/<filename>.csv`.

    Parameters
    ----------
    bucket : str

    dataset : str

    key : str

    Returns
    -------
    (str, dict)
        A pair, where the first item is the filename
        (in the form "<filename>.csv") of the S3 key, and the second item is a
        dict of headers extracted from
        the asset.
    """
    total_sent = 0

    logging.info(f"Translating headers for s3://{bucket}/{key}")

    def cb(bytes_sent):
        nonlocal total_sent
        total_sent += bytes_sent
        logging.debug(
            "extract_header: {}/{} -- transferred {} bytes".format(
                bucket, key, total_sent
            )
        )

    if LOCAL_DATA_DIR:
        path = os.path.join(LOCAL_DATA_DIR, key)
        logging.debug("Reading header {} locally".format(path))

        filename = as_filename(key)
        # Create CSV readers and writers from the temp file objects:
        reader = csv.reader(
            io.TextIOWrapper(open(path, "r+b"), errors="strict"), strict=True
        )

    else:
        # Create temp file objects for raw CSVs:
        raw_csv = NamedTemporaryFile(mode="w+b")

        # Download the raw file from S3 and reset its pointer to the beginning:
        s3_client.download_fileobj(bucket, key, raw_csv, Callback=cb)
        raw_csv.seek(0)

        filename = as_filename(key)

        logging.debug(
            "translate: filename={}, key={}, base={}".format(
                raw_csv.name, key, filename
            )
        )

        # Create CSV readers and writers from the temp file objects:
        reader = csv.reader(io.TextIOWrapper(raw_csv, errors="strict"), strict=True)

    parsed_header = dict()

    # Create temp file objects for parsed CSVs:
    parsed_csv = NamedTemporaryFile(mode="w+b")
    writer = csv.writer(
        io.TextIOWrapper(parsed_csv, errors="strict", write_through=True), strict=True
    )

    # Get the header and parse it:
    header = next(reader)
    parsed_header = parse_header(header)
    output_header = {"props": parsed_header, "model": ""}

    # Create new header row:
    names = list((o["name"] for o in parsed_header))
    writer.writerow(names)

    # Read first row and get model:
    try:
        first_row = next(reader)
    # No data in CSV
    except StopIteration:
        reader_line_count = 1
        # Instead of reading the `label` column in the first row, get the labal
        # from the filename - this should be the same as the label.
        output_header["model"] = filename.split(".")[0]
    else:
        label_index = names.index("~label")
        output_header["model"] = first_row[label_index]
        writer.writerow(first_row)

        reader_line_count = 2
        for row in reader:
            writer.writerow(row)
            reader_line_count += 1

    if not LOCAL_DATA_DIR:
        logging.debug("Read {} lines from {}".format(reader_line_count, key))

        # TODO tidy this
        # Get the destination key, preserving the 'records' and 'relationships' directories
        if remap_ids:
            suffix = remove_prefix(key, remapped_key(dataset, ""))
        else:
            suffix = remove_prefix(key, raw_key(dataset, ""))
        upload_parsed_key = parsed_key(dataset, suffix)

        # Reset the file pointer to the beginning for the parsed csv file
        # so the entire contents can be read out:
        parsed_csv.seek(0)

        logging.debug("=> upload_parsed_key = {}".format(upload_parsed_key))
        s3_client.upload_fileobj(parsed_csv, bucket, upload_parsed_key, Callback=cb)

    return (path_in_dataset(dataset, key), output_header)


def parse_auth_context(bucket, dataset):
    """
    Move the auth context to the parsed bucket.
    """
    s3_client.copy(
        {"Bucket": bucket, "Key": raw_key(dataset, "DatasetInfo.json")},
        bucket,
        parsed_key(dataset, "DatasetInfo.json"),
    )


def get_auth_context(bucket, dataset):
    """
    Get the organization, dataset, and owner information from DatasetInfo.json
    """
    data = fetch_json(bucket, dataset, "DatasetInfo.json")

    return AuthContext(
        organization_id=data["organization_id"],
        organization_node_id=data["organization_node_id"],
        dataset_id=data["dataset_id"],
        dataset_node_id=data["dataset_node_id"],
        user_node_id=data["user_node_id"],
    )


def create_organization_and_dataset(db: PartitionedDatabase):
    with db.transaction() as tx:
        db.db.initialize_organization_and_dataset(
            tx,
            organization_id=auth_context.organization_id,
            dataset_id=auth_context.dataset_id,
            organization_node_id=auth_context.organization_node_id,
            dataset_node_id=auth_context.dataset_node_id,
        )


def check_indexes(db: Database):
    with session(driver) as run:
        indexes = list(run("CALL db.indexes"))

        expected_count = len(index.INDEXES) + len(index.CONSTRAINTS)
        if len(indexes) < expected_count:
            logging.info(
                f"Expected {expected_count} indexes, but only found {len(indexes)}: {indexes} - trying to create indexes"
            )
            index.setup(db)


def run_smoke_test(db, bucket, dataset):
    try:
        dataset_sample = fetch_json(bucket, dataset, "DatasetSample.json", type="test")
    except Exception as e:  # noqa: E722
        raise Exception(
            f"Couldn't find dataset sample {bucket}/{dataset}/test/DatasetSample.json: {str(e)}"
        )

    with db.transaction() as tx:

        for entry in dataset_sample["records"]:

            # sample source record:
            source_record = entry["record"]

            # sample related records (relationship, record):
            for related_to in entry["related_to"]:
                relationship_name = related_to["relationship_name"]
                target_record = related_to["record"]
                target_model_name = related_to["model"]["name"]

                logging.info(
                    f"checking ({source_record})-[{relationship_name}-({target_record}:Model<'{target_model_name}'>)"
                )

                r = db.get_record_tx(tx, RecordId(source_record))

                # Check: originating record exists:
                assert r is not None and str(r.id) == source_record

                connection_map = defaultdict(set)

                for r_relationship, r_record in db.get_related_records_tx(
                    tx, r.id, target_model_name, limit=100000
                ):
                    connection_map[str(r_relationship.name)].add(str(r_record.id))

                assert (
                    relationship_name in connection_map
                    and target_record in connection_map[relationship_name]
                )


def random_uuid(old_id):
    return str(uuid.uuid4())


def remap_ids_in_raw_files(bucket, dataset, generate_new_id):

    id_cache = {}

    for (_, _, key) in raw_files(bucket, dataset):
        if key == raw_key(dataset, "relation_types.csv"):
            remap_csv(
                bucket,
                dataset,
                key,
                ["~id", "$to:String", "$from:String"],
                id_cache,
                generate_new_id,
            )

        elif key == raw_key(dataset, "linkedProperties.csv"):
            remap_csv(
                bucket,
                dataset,
                key,
                ["~id", "~from", "~to", "$schemaLinkedPropertyId:String"],
                id_cache,
                generate_new_id,
            )

        elif key.startswith(raw_key(dataset, "relationships/")):
            remap_csv(
                bucket,
                dataset,
                key,
                ["~id", "~from", "~to", "$schemaRelationshipId:String"],
                id_cache,
                generate_new_id,
            )

        elif (
            key.startswith(raw_key(dataset, "records/"))
            or key == raw_key(dataset, "proxies.csv")
            or key == raw_key(dataset, "proxy_model.csv")
            or key == raw_key(dataset, "models.csv")
            or key == raw_key(dataset, "properties.csv")
        ):
            remap_csv(bucket, dataset, key, ["~id"], id_cache, generate_new_id)

        elif (
            key == raw_key(dataset, "conceptEdges.csv")
            or key == raw_key(dataset, "schemaLinkedProperties.csv")
            or key == raw_key(dataset, "schemaRelations.csv")
            or key == raw_key(dataset, "propertyEdges.csv")
        ):
            remap_csv(
                bucket, dataset, key, ["~id", "~from", "~to"], id_cache, generate_new_id
            )

        else:
            raise Exception(key)

    store_json(bucket, dataset, "remapping.json", id_cache)


def remap_id(old_id, id_cache, generate_new_id):
    old_id = old_id.strip()
    if not old_id:  # model relationship stub with no "from" or "to"
        return ""

    if old_id in id_cache:
        return id_cache[old_id]

    new_id = generate_new_id(old_id)
    id_cache[old_id] = new_id
    return new_id


def remap_csv(bucket, dataset, key, columns, id_cache, generate_new_id):
    """
    For each specified column, remap the ID of the column to a new UUID. This
    mapping is cached and reused in all other places the old UUID appears.
    """
    # Create temp file objects for raw CSVs:
    raw_csv = NamedTemporaryFile(mode="w+b")
    s3_client.download_fileobj(bucket, key, raw_csv)
    raw_csv.seek(0)
    raw_reader = csv.DictReader(io.TextIOWrapper(raw_csv, errors="strict"), strict=True)

    # Remapped temp file and writer
    remapped_csv = NamedTemporaryFile(mode="w+b")

    remapped_writer = csv.DictWriter(
        io.TextIOWrapper(remapped_csv, errors="strict", write_through=True),
        strict=True,
        fieldnames=raw_reader.fieldnames,
    )
    remapped_writer.writeheader()

    for line in raw_reader:
        for column in columns:
            if column in line:
                line[column] = remap_id(line[column], id_cache, generate_new_id)

        remapped_writer.writerow(line)

    suffix = remove_prefix(key, raw_key(dataset, ""))
    upload_remapped_key = remapped_key(dataset, suffix)

    # Reset the file pointer to the beginning for the remapped csv file
    # so the entire contents can be read out:
    remapped_csv.seek(0)

    logging.info("=> upload_remapped_key = {}".format(upload_remapped_key))
    s3_client.upload_fileobj(remapped_csv, bucket, upload_remapped_key)


def load(
    dataset: str,
    bucket: str,
    db: Optional[PartitionedDatabase] = None,
    use_cache: bool = False,
    dry_run: bool = False,
    cutover: bool = False,
    remove_existing: bool = False,
    config: Optional[Config] = None,
    statistics: bool = False,
    smoke_test: bool = True,
    remap_ids: bool = False,
    generate_new_id=random_uuid,
):
    global DRY_RUN_ENABLED, PRODUCE_STATISTICS, driver, s3_client, auth_context

    assert not (use_cache and remap_ids)

    if config is None:
        config = Config()

    logging.basicConfig(level=config.log_level or "INFO")

    if dataset.endswith("/"):
        dataset = dataset[:-1]

    if dataset.startswith("/"):
        dataset = dataset[1:]

    s3_client = boto3.client("s3")

    DRY_RUN_ENABLED = dry_run
    PRODUCE_STATISTICS = statistics

    logging.info(f"Importing dataset s3://{bucket}/{dataset} into {config.neo4j_url}")

    logging.info("Getting auth context")
    if use_cache:
        auth_context = get_auth_context(bucket, dataset)
    else:
        parse_auth_context(bucket, dataset)
        auth_context = get_auth_context(bucket, dataset)

    if db is None:
        db = PartitionedDatabase(
            db=Database.from_config(config),
            organization_id=OrganizationId(auth_context.organization_id),
            dataset_id=DatasetId(auth_context.dataset_id),
            user_id=auth_context.user_node_id,
            organization_node_id=auth_context.organization_node_id,
            dataset_node_id=auth_context.dataset_node_id,
        )
        close_db = True
    else:
        close_db = False

    driver = db.db.driver

    check_indexes(db.db)

    if remove_existing:
        logging.info("Removing existing data")
        db.delete_dataset()
    elif len(db.get_models()) > 0:
        raise Exception(
            "Dataset has already been imported. Rerun with `--remove-existing` to delete existing data"
        )

    if use_cache:
        logging.info("fetching cached headers")
        parsed_headers = fetch_json(bucket, dataset, "headers.json")
    else:
        logging.info("fetching sources")

        # Remap IDs in the PPMI datasets to unique UUIDs
        if remap_ids:
            remap_ids_in_raw_files(bucket, dataset, generate_new_id)
            all_files = remapped_files(bucket, dataset)
            parsed_headers = dict(extract_header(*f, remap_ids=True) for f in all_files)
        else:
            # Reformat input files to match Neo4J specs
            all_files = raw_files(bucket, dataset)
            parsed_headers = dict(extract_header(*f) for f in all_files)

        logging.info("caching headers")
        store_json(bucket, dataset, "headers.json", parsed_headers)

    logging.info("Creating Organization and Dataset entities")
    create_organization_and_dataset(db)

    import_stats = {}

    logging.info("Creating models")
    import_stats.update(create_models(bucket, dataset))

    logging.info("Creating model properties")
    import_stats.update(create_model_properties(bucket, dataset))

    logging.info("Creating model relationships")
    (
        proxy_model_relationships,
        canonical_relationships,
        canonical_relationship_remapping,
        stats,
    ) = create_model_relationships(bucket, dataset)
    import_stats.update(stats)

    logging.info("Creating records")
    import_stats.update(create_records(bucket, dataset, parsed_headers))

    logging.info("Creating relationships")
    import_stats.update(
        create_record_relationships(
            db,
            bucket,
            dataset,
            proxy_model_relationships,
            canonical_relationships,
            canonical_relationship_remapping,
        )
    )

    logging.info("Creating proxy relationships")
    import_stats.update(
        create_proxy_relationships(bucket, dataset, proxy_model_relationships)
    )

    logging.info("Creating model linked properties")
    import_stats.update(create_model_linked_properties(bucket, dataset))

    logging.info("Creating record linked properties")
    import_stats.update(create_record_linked_properties(bucket, dataset, db))

    if PRODUCE_STATISTICS:
        logging.info("Writing statistics.json")
        with open("statistics.json", "w") as statistics_file:
            json.dump(import_stats, statistics_file, indent=4)

    if smoke_test:
        logging.info("Running smoke test sanity check")
        run_smoke_test(db, bucket, dataset)
        logging.info("Running smoke test sanity check - PASSED")

    logging.info("--- DONE ---")

    if close_db and driver:
        driver.close()
