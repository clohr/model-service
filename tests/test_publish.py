import csv
import dataclasses
import datetime
import io
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import UUID, uuid4

import boto3
import pytest
from moto import mock_s3

from publish.config import PublishConfig
from publish.models import (
    ExportedGraphManifests,
    ExportGraphSchema,
    FileManifest,
    ModelService,
)
from publish.publish import publish_dataset, read_file_manifests, write_graph_manifests
from server.models import ModelProperty
from server.models import datatypes as dt


def to_utf8(encoded):
    try:
        return str(encoded, "utf-8")
    except:
        return encoded


@dataclass(frozen=True)
class CSV:
    rows: List[Dict[str, str]]
    size: int


@dataclass(frozen=True)
class JSON:
    content: Any
    size: int


@pytest.fixture(scope="session")
def dataset_id():
    return 10


def sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Sort the rows of a `CSV`.
    """
    return sorted(rows, key=lambda r: list(r.values()))


@pytest.fixture(scope="function")
def s3(aws_credentials):
    with mock_s3():
        yield boto3.client("s3", region_name="us-east-1")


@pytest.fixture(scope="session")
def config(dataset_id):
    return PublishConfig("test-embargo-bucket", f"versioned/{dataset_id}")


@pytest.fixture(scope="function")
def read_csv(s3, config):
    """
    Read a CSV file from S3
    """

    def func(key) -> CSV:
        resp = s3.get_object(Bucket=config.s3_bucket, Key=key)
        reader = csv.DictReader(io.StringIO(to_utf8(resp["Body"].read())))
        return CSV(rows=[row for row in reader], size=resp["ContentLength"])

    return func


@pytest.fixture(scope="function")
def read_json(s3, config):
    """
    Read a JSON file from S3
    """

    def func(key) -> JSON:
        resp = s3.get_object(Bucket=config.s3_bucket, Key=key)
        content = json.loads(to_utf8(resp["Body"].read()))
        return JSON(content=content, size=resp["ContentLength"])

    return func


@pytest.fixture(scope="function")
def metadata_key(config):
    """
    Build a key to a file in the 'metadata' directory of the published dataset.
    """

    def func(filename) -> str:
        return f"{config.s3_publish_key}/metadata/{filename}"

    return func


def test_publish(
    s3,
    dataset_id,
    partitioned_db,
    sample_patient_db,
    config,
    read_csv,
    read_json,
    metadata_key,
):

    # Helpers
    # ==========================================================================

    def id_of(record_name):
        return str(sample_patient_db["records"][record_name].id)

    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    # Setup graph - add more data to the patient DB.
    # ==========================================================================

    # Add a linked property
    best_friend = partitioned_db.create_model_relationship(
        from_model=sample_patient_db["models"]["patient"],
        name="best_friend",
        to_model=sample_patient_db["models"]["patient"],
        display_name="Best friend",
        one_to_many=False,
    )
    partitioned_db.create_record_relationship(id_of("bob"), best_friend, id_of("alice"))

    # Alice has a package proxy
    partitioned_db.create_package_proxy(
        id_of("alice"), package_id=1234, package_node_id="N:package:1234"
    )

    # Bob also has a package proxy. However, this package no longer exists in
    # API. The exporter needs to ignore it.
    # TODO: https://app.clickup.com/t/2c3ec9
    partitioned_db.create_package_proxy(
        id_of("bob"), package_id=4567, package_node_id="N:package:4567"
    )

    # Add another relationship named "attends" The relationship instances for
    # this relationship need to be exported in the same CSV file as the
    # (patient)-[attends]->(visit) relationships, but have a distinct entry in
    # the graph schema.
    event = partitioned_db.create_model("event", display_name="Event", description="")
    partitioned_db.update_properties(
        event,
        ModelProperty(
            name="name", display_name="Name", data_type="String", model_title=True
        ),
    )
    attends = partitioned_db.create_model_relationship(
        sample_patient_db["models"]["patient"], "attends", event
    )
    birthday = partitioned_db.create_records(event, [{"name": "Birthday"}])[0]
    partitioned_db.create_record_relationship(id_of("alice"), attends, birthday)

    # These are the file manifests provided by `discover-publish`
    file_manifests = [
        FileManifest(
            source_file_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            path="files/pkg1/file1.txt",
            size=2293,
            file_type="TEXT",
            source_package_id="N:package:1234",
        ),
        FileManifest(
            source_file_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            path="files/pkg1/file2.csv",
            size=234443,
            file_type="CSV",
            source_package_id="N:package:1234",
        ),
    ]

    # Publish dataset
    # ==========================================================================
    graph_manifests = publish_dataset(
        partitioned_db, s3, config, file_manifests=file_manifests
    )

    for o in s3.list_objects(Bucket=config.s3_bucket).get("Contents", []):
        # Don't export a CSV for the best_friend linked property.
        assert (
            o["Key"] != f"versioned/{dataset_id}/metadata/relationships/best_friend.csv"
        )

    # Check graph schema
    # ==========================================================================

    schema_json = read_json(metadata_key("schema.json"))

    assert sorted(schema_json.content["models"], key=lambda m: m["name"]) == [
        {
            "name": "event",
            "displayName": "Event",
            "description": "",
            "file": "records/event.csv",
            "properties": [
                {
                    "name": "name",
                    "displayName": "Name",
                    "description": "",
                    "dataType": {"type": "String"},
                }
            ],
        },
        {
            "name": "file",
            "displayName": "File",
            "description": "A file in the dataset",
            "file": "records/file.csv",
            "properties": [
                {
                    "name": "path",
                    "displayName": "Path",
                    "description": "The path to the file from the root of the dataset",
                    "dataType": {"type": "String"},
                }
                # TODO: add sourcePackageId (enhancement)
            ],
        },
        {
            "name": "medication",
            "displayName": "Medication",
            "description": "a medication",
            "file": "records/medication.csv",
            "properties": [
                {
                    "name": "name",
                    "displayName": "Name",
                    "description": "",
                    "dataType": {"type": "String"},
                }
            ],
        },
        {
            "name": "patient",
            "displayName": "Patient",
            "description": "a person",
            "file": "records/patient.csv",
            "properties": [
                {
                    "name": "name",
                    "displayName": "Name",
                    "description": "",
                    "dataType": {"type": "String"},
                },
                {
                    "name": "age",
                    "displayName": "Age",
                    "description": "",
                    "dataType": {"type": "Long"},
                },
                {
                    "name": "best_friend",
                    "displayName": "Best friend",
                    "description": "",
                    "dataType": {
                        "type": "Model",
                        "to": "patient",
                        "file": "records/patient.csv",
                    },
                },
            ],
        },
        {
            "name": "visit",
            "displayName": "Visit",
            "description": "a visit",
            "file": "records/visit.csv",
            "properties": [
                {
                    "name": "day",
                    "displayName": "Day",
                    "description": "",
                    "dataType": {"type": "String"},
                }
            ],
        },
    ]

    assert sorted(
        schema_json.content["relationships"], key=lambda r: (r["from"], r["to"])
    ) == sorted(
        [
            {
                "name": "attends",
                "from": "patient",
                "to": "visit",
                "file": "relationships/attends.csv",
            },
            {
                "name": "attends",
                "from": "patient",
                "to": "event",
                "file": "relationships/attends.csv",
            },
            {
                "name": "belongs_to",
                "from": "",
                "to": "",
                "file": "relationships/belongs_to.csv",
            },
            {
                "name": "prescribed",
                "from": "visit",
                "to": "medication",
                "file": "relationships/prescribed.csv",
            },
        ],
        key=lambda r: (r["from"], r["to"]),
    )

    # Check records
    # ==========================================================================

    patient_csv = read_csv(metadata_key("records/patient.csv"))
    assert sort_rows(patient_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "id": id_of("alice"),
                    "name": "Alice",
                    "age": "34",
                    "best_friend": None,
                    "best_friend:display": None,
                }
            ),
            OrderedDict(
                {
                    "id": id_of("bob"),
                    "name": "Bob",
                    "age": "20",
                    "best_friend": id_of("alice"),
                    "best_friend:display": "Alice",
                }
            ),
        ]
    )

    visit_csv = read_csv(metadata_key("records/visit.csv"))
    assert sort_rows(visit_csv.rows) == sort_rows(
        [
            OrderedDict({"id": id_of("monday"), "day": "Monday"}),
            OrderedDict({"id": id_of("tuesday"), "day": "Tuesday"}),
        ]
    )

    medication_csv = read_csv(metadata_key("records/medication.csv"))
    assert sort_rows(medication_csv.rows) == sort_rows(
        [
            OrderedDict({"id": id_of("aspirin"), "name": "Aspirin"}),
            OrderedDict({"id": id_of("motrin"), "name": "Motrin"}),
            OrderedDict({"id": id_of("tylenol"), "name": "Tylenol"}),
        ]
    )

    event_csv = read_csv(metadata_key("records/event.csv"))
    assert event_csv.rows == [{"id": str(birthday.id), "name": "Birthday"}]

    # Check relationships
    # ==========================================================================

    attends_csv = read_csv(metadata_key("relationships/attends.csv"))
    assert sort_rows(attends_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "from": id_of("alice"),
                    "to": id_of("monday"),
                    "relationship": "attends",
                }
            ),
            OrderedDict(
                {
                    "from": id_of("bob"),
                    "to": id_of("tuesday"),
                    "relationship": "attends",
                }
            ),
            # Contains relationships from multiple model relationships
            OrderedDict(
                {
                    "from": id_of("alice"),
                    "to": str(birthday.id),
                    "relationship": "attends",
                }
            ),
        ]
    )

    prescribed_csv = read_csv(metadata_key("relationships/prescribed.csv"))
    assert sort_rows(prescribed_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "from": id_of("monday"),
                    "to": id_of("aspirin"),
                    "relationship": "prescribed",
                }
            ),
            OrderedDict(
                {
                    "from": id_of("tuesday"),
                    "to": id_of("aspirin"),
                    "relationship": "prescribed",
                }
            ),
            OrderedDict(
                {
                    "from": id_of("tuesday"),
                    "to": id_of("tylenol"),
                    "relationship": "prescribed",
                }
            ),
        ]
    )

    # Check proxy packages
    # ==========================================================================

    file_csv = read_csv(metadata_key("records/file.csv"))
    assert sort_rows(file_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "sourceFileId": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "path": "files/pkg1/file1.txt",
                    "sourcePackageId": "N:package:1234",
                }
            ),
            OrderedDict(
                {
                    "sourceFileId": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "path": "files/pkg1/file2.csv",
                    "sourcePackageId": "N:package:1234",
                }
            ),
        ]
    )

    # Check proxy package relationships
    # ==========================================================================

    belongs_to_csv = read_csv(metadata_key("relationships/belongs_to.csv"))
    assert sort_rows(belongs_to_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "from": id_of("alice"),
                    "to": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "relationship": "belongs_to",
                }
            ),
            OrderedDict(
                {
                    "from": id_of("alice"),
                    "to": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "relationship": "belongs_to",
                }
            ),
        ]
    )

    # Check file manifest output
    # ==========================================================================

    # Check that version IDs exist for all manifests, then remove them to allow
    # for comparison:
    for manifest in graph_manifests:
        assert manifest.version_id is not None

    graph_manifests = [
        dataclasses.replace(manifest, version_id=None) for manifest in graph_manifests
    ]

    assert sorted(graph_manifests) == sorted(
        [
            FileManifest(
                path="metadata/schema.json", file_type="Json", size=schema_json.size
            ),
            FileManifest(
                path="metadata/records/event.csv", file_type="CSV", size=event_csv.size
            ),
            FileManifest(
                path="metadata/records/file.csv", file_type="CSV", size=file_csv.size
            ),
            FileManifest(
                path="metadata/records/medication.csv",
                file_type="CSV",
                size=medication_csv.size,
            ),
            FileManifest(
                path="metadata/records/patient.csv",
                file_type="CSV",
                size=patient_csv.size,
            ),
            FileManifest(
                path="metadata/records/visit.csv", file_type="CSV", size=visit_csv.size
            ),
            FileManifest(
                path="metadata/relationships/attends.csv",
                file_type="CSV",
                size=attends_csv.size,
            ),
            FileManifest(
                path="metadata/relationships/prescribed.csv",
                file_type="CSV",
                size=prescribed_csv.size,
            ),
            FileManifest(
                path="metadata/relationships/belongs_to.csv",
                file_type="CSV",
                size=belongs_to_csv.size,
            ),
        ]
    )


def test_record_value_serialization(s3, config, read_csv, metadata_key, partitioned_db):
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    patient = partitioned_db.create_model("patient", "Patient")
    partitioned_db.update_properties(
        patient,
        ModelProperty(
            name="string",
            display_name="String",
            data_type=dt.String(),
            model_title=True,
        ),
        ModelProperty(name="boolean", display_name="Boolean", data_type=dt.Boolean()),
        ModelProperty(name="long", display_name="Long", data_type=dt.Long()),
        ModelProperty(name="double", display_name="Double", data_type=dt.Double()),
        ModelProperty(name="date", display_name="Date", data_type=dt.Date()),
        ModelProperty(name="optional", display_name="Optional", data_type=dt.String()),
        ModelProperty(
            name="string_array",
            display_name="String Array",
            data_type=dt.Array(items=dt.String()),
        ),
        ModelProperty(
            name="boolean_array",
            display_name="Boolean Array",
            data_type=dt.Array(items=dt.Boolean()),
        ),
        ModelProperty(
            name="long_array",
            display_name="Long Array",
            data_type=dt.Array(items=dt.Long()),
        ),
        ModelProperty(
            name="double_array",
            display_name="Double Array",
            data_type=dt.Array(items=dt.Double()),
        ),
        ModelProperty(
            name="date_array",
            display_name="Date Array",
            data_type=dt.Array(items=dt.Date()),
        ),
    )
    record = partitioned_db.create_records(
        patient,
        [
            {
                "string": 'tricky"char,acter"string',
                "boolean": True,
                "long": 12345,
                "double": 3.14159,
                "date": datetime.datetime(year=2004, month=5, day=5),
                "optional": None,
                "string_array": ["red", "green", "semi;colon"],
                "boolean_array": [True, False],
                "long_array": [1, 2, 3],
                "double_array": [1.1, 2.2, 3.3],
                "date_array": [
                    datetime.datetime(year=2004, month=5, day=5),
                    datetime.datetime(year=2014, month=5, day=16),
                ],
            }
        ],
    )[0]

    publish_dataset(partitioned_db, s3, config, file_manifests=[])

    patient_csv = read_csv(metadata_key("records/patient.csv"))
    assert patient_csv.rows == [
        OrderedDict(
            {
                "id": str(record.id),
                "string": 'tricky"char,acter"string',
                "boolean": "true",
                "long": "12345",
                "double": "3.14159",
                "date": "2004-05-05T00:00:00",
                "optional": "",
                "string_array": "red;green;semi_colon",
                "boolean_array": "true;false",
                "long_array": "1;2;3",
                "double_array": "1.1;2.2;3.3",
                "date_array": "2004-05-05T00:00:00;2014-05-16T00:00:00",
            }
        )
    ]


def test_proxy_relationships_are_merged_with_record_relationships(
    s3, dataset_id, config, read_csv, metadata_key, partitioned_db
):
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    person = partitioned_db.create_model("person", "Person")
    partitioned_db.update_properties(
        person,
        ModelProperty(
            name="name", display_name="String", data_type=dt.String(), model_title=True
        ),
    )

    item = partitioned_db.create_model("item", "Item")
    partitioned_db.update_properties(
        item,
        ModelProperty(
            name="name", display_name="String", data_type=dt.String(), model_title=True
        ),
    )

    # This relationship uses the default "belongs_to" package proxy relationship,
    # and should be exported in the same CSV file.
    item_belongs_to_person = partitioned_db.create_model_relationship(
        item, "belongs_to", person, one_to_many=True
    )

    person_likes_person = partitioned_db.create_model_relationship(
        person, "likes", person, one_to_many=True
    )

    alice = partitioned_db.create_record(person, {"name": "Alice"})

    bob = partitioned_db.create_record(person, {"name": "Bob"})

    laptop = partitioned_db.create_record(item, {"name": "Laptop"})

    partitioned_db.create_record_relationship(alice, person_likes_person, bob)
    partitioned_db.create_record_relationship(laptop, item_belongs_to_person, alice)

    # Package proxy using default `belongs_to` relationship
    partitioned_db.create_package_proxy(
        alice, package_id=1234, package_node_id="N:package:1234"
    )

    # Package proxy using a non-standard `likes` relationship
    partitioned_db.create_package_proxy(
        alice,
        package_id=4567,
        package_node_id="N:package:4567",
        legacy_relationship_type="likes",
    )

    file_manifests = [
        FileManifest(
            source_file_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            path=f"versioned/{dataset_id}/files/pkg1/file1.txt",
            size=2293,
            file_type="TEXT",
            source_package_id="N:package:1234",
        ),
        FileManifest(
            source_file_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            path=f"versioned/{dataset_id}/files/pkg1/file2.csv",
            size=234443,
            file_type="CSV",
            source_package_id="N:package:1234",
        ),
        FileManifest(
            source_file_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            path=f"versioned/{dataset_id}/files/pkg2/file3.dcm",
            size=338923,
            file_type="DICOM",
            source_package_id="N:package:4567",
        ),
    ]

    graph_manifests = publish_dataset(
        partitioned_db, s3, config, file_manifests=file_manifests
    )

    assert sorted([m.path for m in graph_manifests]) == [
        "metadata/records/file.csv",
        "metadata/records/item.csv",
        "metadata/records/person.csv",
        "metadata/relationships/belongs_to.csv",
        "metadata/relationships/likes.csv",
        "metadata/schema.json",
    ]

    belongs_to_csv = read_csv(metadata_key("relationships/belongs_to.csv"))
    assert sort_rows(belongs_to_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "from": str(alice.id),
                    "to": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "relationship": "belongs_to",
                }
            ),
            OrderedDict(
                {
                    "from": str(alice.id),
                    "to": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "relationship": "belongs_to",
                }
            ),
            OrderedDict(
                {
                    "from": str(laptop.id),
                    "to": str(alice.id),
                    "relationship": "belongs_to",
                }
            ),
        ]
    )

    likes_csv = read_csv(metadata_key("relationships/likes.csv"))
    assert sort_rows(likes_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "from": str(alice.id),
                    "to": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                    "relationship": "likes",
                }
            ),
            OrderedDict(
                {"from": str(alice.id), "to": str(bob.id), "relationship": "likes"}
            ),
        ]
    )


def test_publish_proxy_record_csv_when_no_proxy_relationships_exist(
    s3, dataset_id, config, metadata_key, partitioned_db, read_json
):
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    file_manifests = [
        FileManifest(
            source_file_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            path=f"versioned/{dataset_id}/files/pkg1/file1.txt",
            size=2293,
            file_type="TEXT",
            source_package_id="N:package:1234",
        ),
        FileManifest(
            source_file_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            path=f"versioned/{dataset_id}/files/pkg2/file3.dcm",
            size=338923,
            file_type="DICOM",
            source_package_id="N:package:4567",
        ),
    ]

    graph_manifests = publish_dataset(
        partitioned_db, s3, config, file_manifests=file_manifests
    )
    assert sorted([m.path for m in graph_manifests]) == [
        "metadata/records/file.csv",
        "metadata/schema.json",
    ]

    schema_json = read_json(metadata_key("schema.json"))
    assert [m["name"] for m in schema_json.content["models"]] == ["file"]
    assert len(schema_json.content["relationships"]) == 0


def test_do_not_publish_proxy_record_csv_when_no_source_files_exist(
    s3, config, metadata_key, partitioned_db, read_json
):
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    graph_manifests = publish_dataset(partitioned_db, s3, config, file_manifests=[])
    assert sorted([m.path for m in graph_manifests]) == ["metadata/schema.json"]

    schema_json = read_json(metadata_key("schema.json"))
    assert len(schema_json.content["models"]) == 0
    assert len(schema_json.content["relationships"]) == 0


def test_publish_thousands_of_records(
    s3, config, read_csv, metadata_key, partitioned_db
):
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    patient = partitioned_db.create_model("patient", "Patient")
    partitioned_db.create_records(patient, [{} for i in range(2000)])

    publish_dataset(partitioned_db, s3, config, file_manifests=[])

    patient_csv = read_csv(metadata_key("records/patient.csv"))
    assert len(patient_csv.rows) == 2000


def test_publish_linked_properties_with_no_index(
    s3, config, read_csv, read_json, metadata_key, partitioned_db
):
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    gene = partitioned_db.create_model("gene", "Gene")
    partitioned_db.update_properties(
        gene,
        ModelProperty(
            "name", "name", data_type=dt.String(), model_title=True, required=True
        ),
    )
    regulates = partitioned_db.create_model_relationship(
        gene, "regulates", gene, one_to_many=False, index=None
    )
    interacts = partitioned_db.create_model_relationship(
        gene, "interacts", gene, one_to_many=False, index=1
    )

    yy1 = partitioned_db.create_record(gene, {"name": "YY1"})
    pepd = partitioned_db.create_record(gene, {"name": "PEPD"})
    gmpr2 = partitioned_db.create_record(gene, {"name": "GMPR2"})

    partitioned_db.create_record_relationship(yy1, regulates, gmpr2)
    partitioned_db.create_record_relationship(yy1, interacts, pepd)

    publish_dataset(partitioned_db, s3, config, file_manifests=[])

    schema_json = read_json(metadata_key("schema.json"))
    assert schema_json.content["models"][0]["properties"] == [
        {
            "name": "name",
            "displayName": "name",
            "description": "",
            "dataType": {"type": "String"},
        },
        {
            "name": "interacts",
            "displayName": "interacts",
            "description": "",
            "dataType": {"type": "Model", "to": "gene", "file": "records/gene.csv"},
        },
        {
            "name": "regulates",
            "displayName": "regulates",
            "description": "",
            "dataType": {"type": "Model", "to": "gene", "file": "records/gene.csv"},
        },
    ]

    gene_csv = read_csv(metadata_key("records/gene.csv"))
    assert sort_rows(gene_csv.rows) == sort_rows(
        [
            OrderedDict(
                {
                    "id": str(yy1.id),
                    "name": "YY1",
                    "interacts": str(pepd.id),
                    "interacts:display": "PEPD",
                    "regulates": str(gmpr2.id),
                    "regulates:display": "GMPR2",
                }
            ),
            OrderedDict(
                {
                    "id": str(pepd.id),
                    "name": "PEPD",
                    "interacts": None,
                    "interacts:display": None,
                    "regulates": None,
                    "regulates:display": None,
                }
            ),
            OrderedDict(
                {
                    "id": str(gmpr2.id),
                    "name": "GMPR2",
                    "interacts": None,
                    "interacts:display": None,
                    "regulates": None,
                    "regulates:display": None,
                }
            ),
        ]
    )


def test_read_file_manifests(s3, dataset_id):
    config = PublishConfig("test-publish-bucket", f"versioned/{dataset_id}")
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    version_id = str(uuid4())
    body = f"""
        {{
          "externalIdToPackagePath" : {{
          }},
          "packageManifests" : [
            {{
              "path" : "files/test.txt",
              "size" : 43649,
              "fileType" : "TEXT",
              "sourceFileId" : "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
              "sourcePackageId" : "N:package:1234",
              "versionId": "{version_id}"
            }}
          ],
          "bannerKey" : "banner.jpg",
          "bannerManifest" : {{
            "path" : "banner.jpg",
            "size" : 43649,
            "fileType" : "JPEG"
          }},
          "readmeKey" : "readme.md",
          "readmeManifest" : {{
            "path" : "readme.md",
            "size" : 15,
            "fileType" : "Markdown"
          }}
        }}
    """

    # This would be created by `discover-publish`
    s3.put_object(
        Bucket=config.s3_bucket,
        Key=os.path.join(config.s3_publish_key, "publish.json"),
        Body=body,
    )

    assert read_file_manifests(s3, config) == [
        FileManifest(
            path="files/test.txt",
            size=43649,
            file_type="TEXT",
            source_file_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            source_package_id="N:package:1234",
            version_id=version_id,
        )
    ]


def test_write_graph_manifests(s3, dataset_id):
    config = PublishConfig("test-publish-bucket", f"versioned/{dataset_id}")
    s3.create_bucket(Bucket=config.s3_bucket)
    s3.put_bucket_versioning(
        Bucket=config.s3_bucket, VersioningConfiguration={"Status": "Enabled"}
    )

    graph_manifests = [
        FileManifest(path="metadata/schema.json", file_type="Json", size=233),
        FileManifest(path="metadata/records/file.csv", file_type="CSV", size=533),
    ]

    write_graph_manifests(s3, config, graph_manifests)

    assert (
        ExportedGraphManifests.schema()
        .loads(
            s3.get_object(
                Bucket=config.s3_bucket, Key=f"versioned/{dataset_id}/graph.json"
            )["Body"].read()
        )
        .manifests
        == graph_manifests
    )
