import csv
import json
import logging
import os
import sys
from collections import Sequence, Set, defaultdict
from contextlib import contextmanager
from datetime import datetime
from itertools import chain
from tempfile import NamedTemporaryFile
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, cast
from uuid import UUID

import neotime  # type: ignore
from neo4j import Transaction  # type: ignore

from server.auth import AuthContext
from server.config import Config
from server.db import Database, PartitionedDatabase, labels
from server.models import (
    DatasetId,
    Model,
    ModelProperty,
    ModelRelationship,
    OrganizationId,
    PackageProxy,
    Record,
    RecordRelationship,
    RecordStub,
    RelationshipName,
)
from server.models import datatypes as dt
from server.models import normalize_relationship_type

from .config import PublishConfig
from .models import (
    ExportedDatasetLocation,
    ExportedGraphManifests,
    ExportGraphSchema,
    ExportModel,
    ExportModelRelationship,
    ExportProperty,
    FileManifest,
    LinkedModelDataType,
    ModelService,
    OutputFile,
    PackageProxyRelationship,
)

log = logging.getLogger(__name__)


PUBLISH_ASSETS_FILE = OutputFile.json("publish")

GRAPH_ASSETS_FILE = OutputFile.json("graph")

DATASET_LOCATION_FILE = OutputFile.json("service")

PROXY_FILE_MODEL_NAME = "file"

METADATA = "metadata"


def publish_dataset(
    db: PartitionedDatabase,
    s3,
    config: PublishConfig,
    file_manifests: List[FileManifest],
) -> List[FileManifest]:
    """
    Publish the dataset provided by the partitioned database view in `db`.

    Returns a list of file manifests for each file written to S3.
    """
    graph_manifests: List[FileManifest] = []

    with db.transaction() as tx:

        # 0) Get all existing proxy relationships. We'll need these in a few places.
        # ======================================================================
        proxies_by_relationship: Dict[
            RelationshipName, List[PackageProxyRelationship]
        ] = defaultdict(list)
        for pp in package_proxy_relationships(db, tx, config, s3, file_manifests):
            proxies_by_relationship[pp.relationship].append(pp)

        # 1) Publish graph schema
        # ======================================================================
        schema, schema_manifest = publish_schema(
            db, tx, config, s3, file_manifests, list(proxies_by_relationship.keys())
        )
        graph_manifests.append(schema_manifest)

        # 2) Publish record CSVs
        # ======================================================================
        for m in schema.models:
            if m.model:
                graph_manifests.append(
                    publish_records_of_model(db, tx, m.model, config, s3)
                )

        # 3) Publish source file CSV
        # ======================================================================
        if len(file_manifests) > 0:
            graph_manifests.append(
                publish_package_proxy_files(db, tx, config, s3, file_manifests)
            )

        # 4) Publish relationship CSVs
        # ======================================================================

        # All relationships with the same name need to go in the same CSV file,
        # along with proxy relationships with the same name.
        relationships_by_name: Dict[
            RelationshipName, List[ModelRelationship]
        ] = defaultdict(list)
        for r in schema.relationships:
            if not r.is_proxy_relationship():
                assert r.relationship is not None
                relationships_by_name[r.name].append(r.relationship)

        for relationship_name in set(
            chain(relationships_by_name.keys(), proxies_by_relationship.keys())
        ):
            graph_manifests.append(
                publish_relationships(
                    db,
                    tx,
                    relationship_name,
                    relationships_by_name[relationship_name],
                    proxies_by_relationship[relationship_name],
                    config,
                    s3,
                )
            )

    return graph_manifests


def read_file_manifests(s3, config: PublishConfig) -> List[FileManifest]:
    """
    The initial invocation of `discover-publish` writes a file containing manifests
    for all source files.
    """
    log.info("Reading source file manifests")

    body = s3.get_object(
        Bucket=config.s3_bucket,
        Key=str(PUBLISH_ASSETS_FILE.with_prefix(config.s3_publish_key)),
    )["Body"].read()

    return FileManifest.schema().load(json.loads(body)["packageManifests"], many=True)


def write_graph_manifests(
    s3, config: PublishConfig, graph_manifests: List[FileManifest]
) -> None:
    """
    Write all produced file manifests to S3 for `discover-publish` to collate.
    """
    log.info("Writing graph file manifests")

    graph_manifest_file = GRAPH_ASSETS_FILE.with_prefix(config.s3_publish_key)

    exported_manifests = ExportedGraphManifests(graph_manifests)

    s3.put_object(
        Bucket=config.s3_bucket,
        Key=str(graph_manifest_file),
        Body=exported_manifests.to_json(camel_case=True),
    )


def publish_schema(
    db: PartitionedDatabase,
    tx: Transaction,
    config: PublishConfig,
    s3,
    file_manifests: List[FileManifest],
    proxy_relationship_names: List[RelationshipName],
) -> Tuple[ExportGraphSchema, FileManifest]:
    """
    Export the schema of the partitioned database into a `GraphSchema`
    instance.
    """
    schema_models: List[ExportModel] = []
    schema_relationships: List[ExportModelRelationship] = []

    log.info("Exporting graph schema")

    models: List[Model] = db.get_models_tx(tx)
    model_index: Dict[UUID, Model] = {m.id: m for m in models}

    for m in models:
        log.info(f"Building schema for model '{m.name}'")
        properties: List[ModelProperty] = db.get_properties_tx(tx, m)
        linked_properties: List[ModelRelationship] = list(
            db.get_outgoing_model_relationships_tx(tx, from_model=m, one_to_many=False)
        )

        publish_properties: List[ExportProperty] = [
            ExportProperty.model_property(
                name=p.name,
                display_name=p.display_name,
                description=p.description,
                data_type=p.data_type,
            )
            for p in properties
        ] + [
            ExportProperty.linked_property(
                name=r.name,
                display_name=r.display_name,
                description=r.description,
                data_type=LinkedModelDataType(
                    to=model_index[r.to].name,
                    file=str(OutputFile.csv_for_model(m.name)),
                ),
            )
            for r in sorted(linked_properties, key=lambda l: l.index or sys.maxsize)
        ]

        model = ExportModel(
            model=m,
            name=m.name,
            display_name=m.display_name,
            description=m.description,
            properties=publish_properties,
        )
        schema_models.append(model)

    # If any packages exist in this dataset, add a special-cased "File" model
    if len(file_manifests) > 0:
        log.info(f"Building schema for proxy package model")
        proxy_package_model = ExportModel.package_proxy()

        # TODO: gracefully handle this case to avoid overwriting "files.csv"
        assert not any(m.name == proxy_package_model.name for m in schema_models), (
            f"Cannot export package proxy schema model with name '{proxy_package_model.name}' - "
            f"a model '{m.name}' already exists. See https://app.clickup.com/t/102ndc for issue"
        )
        schema_models.append(proxy_package_model)

    relationships = db.get_outgoing_model_relationships_tx(tx, one_to_many=True)

    for r in relationships:
        log.info(f"Building schema for relationship '{r.name}'")
        relationship = ExportModelRelationship(
            relationship=r,
            name=r.name,
            from_=model_index[r.from_].name,
            to=model_index[r.to].name,
        )
        schema_relationships.append(relationship)

    for p in proxy_relationship_names:
        log.info(f"Building schema for proxy relationship '{p}'")
        relationship = ExportModelRelationship(
            relationship=None, name=p, from_="", to=""
        )
        schema_relationships.append(relationship)

    schema = ExportGraphSchema(models=schema_models, relationships=schema_relationships)

    # Write "schema.json" to S3
    # ======================================================================
    schema_output_file = OutputFile.json_for_schema().with_prefix(
        os.path.join(config.s3_publish_key, METADATA)
    )
    s3.put_object(
        Bucket=config.s3_bucket,
        Key=str(schema_output_file),
        Body=schema.to_json(camel_case=True, pretty_print=True, drop_nulls=True),
    )
    schema_manifest = schema_output_file.with_prefix(METADATA).as_manifest(
        size_of(s3, config.s3_bucket, schema_output_file)
    )

    return schema, schema_manifest


def relationship_headers() -> List[str]:
    """
    Produce headers for a relationship file.
    """
    return ["from", "to", "relationship"]


def record_headers(
    properties: List[ModelProperty], linked_properties: List[ModelRelationship]
) -> List[str]:
    """
    Extract the headers from the properties of a model, perserving ordering.
    """
    return (
        ["id"]
        + [p.name for p in properties]
        + list(
            chain.from_iterable(
                [(lp.name, f"{lp.name}:display") for lp in linked_properties]
            )
        )
    )


def format_value(v) -> str:
    """
    Format a record value to export in a CSV

    This exactly matches the format from the original `concepts-service` export:
    https://github.com/Blackfynn/blackfynn-api/blob/0410ca100fc2732d234eb61d0301f6fbb7057c45/core-models/src/main/scala/com/blackfynn/concepts/Proxy.scala#L89-L104

    TODO: Should we export lists as JSON instead of semicolon-separated strings?
    """
    if v is None:
        return ""
    elif v is True:
        return "true"
    elif v is False:
        return "false"
    elif isinstance(v, str):
        return v
    elif isinstance(v, (Sequence, Set)):
        return ";".join(format_value(w).replace(";", "_") for w in v)
    elif isinstance(v, datetime):
        return v.isoformat()
    elif isinstance(v, neotime.DateTime):
        return v.to_native().isoformat()
    return str(v)


def record_row(
    r: Record,
    properties: List[ModelProperty],
    linked_properties: List[ModelRelationship],
) -> List[str]:
    """
    Return a CSV row for a single record.

    Linked properties are written as two columns, one containing the id, the
    other containing the display name of the related record. The header for the
    display name has a `:display` suffix added.
    """
    # The record ID always comes first:
    row = [str(r.id)]

    # Output all properties for the record according to the ordering of
    # properties supplied in the model property list:
    for p in properties:
        assert p.name in r.values, f"Missing property {p.name} from record {r.id}"
        row.append(format_value(r.values[p.name]))

    for lp in linked_properties:
        linked_record = cast(RecordStub, r.values.get(lp.name))
        if linked_record:
            row.append(str(linked_record.id))
            row.append(linked_record.title or "")

    return row


def publish_records_of_model(
    db: PartitionedDatabase, tx: Transaction, model: Model, config, s3
) -> FileManifest:
    """
    Export the records of a specific model.
    """
    log.info(f"Writing records for model '{model.name}'")

    output_file: OutputFile = OutputFile.csv_for_model(model.name).with_prefix(
        os.path.join(config.s3_publish_key, METADATA)
    )

    model_properties: List[ModelProperty] = db.get_properties_tx(tx, model)

    linked_properties: List[ModelRelationship] = sorted(
        db.get_outgoing_model_relationships_tx(tx, from_model=model, one_to_many=False),
        key=lambda r: r.index or sys.maxsize,
    )

    # Construct the header list for a model:
    headers: List[str] = record_headers(model_properties, linked_properties)

    with s3_csv_writer(s3, config.s3_bucket, str(output_file), headers) as writer:
        for r in db.get_all_records_offset_tx(
            tx=tx,
            model=model,
            embed_linked=True,
            fill_missing=True,
            limit=None,
        ):
            writer.writerow(record_row(r, model_properties, linked_properties))

    return output_file.with_prefix(METADATA).as_manifest(
        size_of(s3, config.s3_bucket, output_file)
    )


def publish_package_proxy_files(
    db: PartitionedDatabase,
    tx: Transaction,
    config,
    s3,
    file_manifests: List[FileManifest],
) -> FileManifest:
    """
    If a relationship exists between a concept and a package that no longer
    exists, ignore the relationship. This is a race condition between API and
    `model-service`.

    If a relationship points to a package, the relationship is expanded into a
    set of relationships between the concept and the source files of the
    package. For example,

        [record] ---> [proxy] -----> [file1]
                              \\ ---> [file2]

    becomes

        [record] -----> [file1]
                 \\ ---> [file2]

    """
    log.info(f"Writing proxy packages")

    file_output_file = OutputFile.csv_for_model(PROXY_FILE_MODEL_NAME).with_prefix(
        os.path.join(config.s3_publish_key, METADATA)
    )

    with s3_csv_writer(
        s3,
        config.s3_bucket,
        str(file_output_file),
        headers=["id", "path", "sourcePackageId"],
    ) as writer:
        for file_manifest in file_manifests:
            if file_manifest.source_package_id:
                writer.writerow(
                    [
                        file_manifest.id,
                        file_manifest.path,
                        file_manifest.source_package_id,
                    ]
                )

    return file_output_file.with_prefix(METADATA).as_manifest(
        size_of(s3, config.s3_bucket, file_output_file)
    )


def package_proxy_relationships(
    db: PartitionedDatabase,
    tx: Transaction,
    config,
    s3,
    file_manifests: List[FileManifest],
) -> Iterator[PackageProxyRelationship]:
    """
    Yield all proxy package relationships in the dataset

    Explodes each proxy package into multiple source files.  If the package no
    longer exists in the dataset, ignore it.
    """

    files_by_package_id: Dict[str, List[FileManifest]] = defaultdict(list)
    for f in file_manifests:
        if f.source_package_id:
            files_by_package_id[f.source_package_id].append(f)

    for pp, record in db.get_all_package_proxies_tx(tx):
        for file_manifest in files_by_package_id.get(pp.package_node_id, []):
            assert file_manifest.id is not None

            yield PackageProxyRelationship(
                from_=record.id, to=file_manifest.id, relationship=pp.relationship_type
            )


def publish_relationships(
    db: PartitionedDatabase,
    tx: Transaction,
    relationship_name: RelationshipName,
    relationships: List[ModelRelationship],
    proxy_relationships: List[PackageProxyRelationship],
    config,
    s3,
) -> FileManifest:
    """
    Export record relationships for all model relationships and package proxy
    relationships with the same name.

    TODO: it would be cleaner to export every model relationship to a distinct
    CSV.  For example, when publishing a graph with

        (patient)-[attends]->(visit)

    and

        (patient)-[attends]->(event)

    relationships, we currently write these relationships to the same
    `attends.csv` file.  If we wrote two separate `patient_attends_visit.csv`
    and `patient_attends_event.csv` files, we would not need to group
    relationships by name, nor add proxy relationships to these CSVs. The CSV
    for proxy relationships would just become `patient_belongs_to_file.csv` for
    every model with proxy packages.
    """
    log.info(f"Writing record relationships for relationship '{relationship_name}'")

    assert all(
        r.name == relationship_name for r in relationships
    ), f"Relationships have different names: {[r.name for r in relationships]}"

    assert all(
        pp.relationship == relationship_name for pp in proxy_relationships
    ), f"Package proxy relationships have different names: {[pp.relationship for pp in proxy_relationships]}"

    output_file: OutputFile = OutputFile.csv_for_relationship(
        relationship_name
    ).with_prefix(os.path.join(config.s3_publish_key, METADATA))

    with s3_csv_writer(
        s3, config.s3_bucket, str(output_file), relationship_headers()
    ) as writer:
        for relationship in relationships:
            for rr in db.get_record_relationships_by_model_tx(tx, relationship):
                writer.writerow([str(rr.from_), str(rr.to), str(rr.name)])

        for pp in proxy_relationships:
            writer.writerow([str(pp.from_), str(pp.to), str(pp.relationship)])

    return output_file.with_prefix(METADATA).as_manifest(
        size_of(s3, config.s3_bucket, output_file)
    )


def size_of(s3, bucket, key) -> int:
    return s3.head_object(Bucket=bucket, Key=str(key))["ContentLength"]


@contextmanager
def s3_csv_writer(s3, bucket, key, headers):
    """
    Context manager for writing a CSV file to S3

    Yields a stdlib `csv.writer` object.
    """
    log.info(f"Writing file '{key}' with headers {headers} to S3")

    # Write output locally
    with NamedTemporaryFile(delete=False, mode="w") as temp_file:
        writer = csv.writer(temp_file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        yield writer

    s3.upload_file(Filename=temp_file.name, Bucket=bucket, Key=str(key))
