import dataclasses
from uuid import UUID, uuid4

import neotime
import pytest
import pytz
from dateutil import parser as iso8601

from loader.import_to_neo4j import load
from server.db import PartitionedDatabase
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
)
from server.models import datatypes as dt
from server.models.legacy import ModelRelationshipStub


"""
These tests are dependent on S3. They should be converted to use localstack, but
the way that Neo4j loads CSV files makes this tricky: the presigned URLs that we
build point to localhost, which does not mean anything inside of the Neo4j
container
"""


@pytest.mark.skip
@pytest.mark.integration
def test_loader(neo4j):
    dataset_id = 29233
    dataset_node_id = "N:dataset:b1154216-d1d7-4484-ad18-81b58fb65484"
    organization_id = 5
    organization_node_id = "N:organization:c905919f-56f5-43ae-9c2a-8d5d542c133b"
    user_id = 114
    user_node_id = "N:user:028058b9-dd8d-4f24-a187-ea56830b379f"

    db = PartitionedDatabase(
        db=neo4j,
        organization_id=OrganizationId(organization_id),
        dataset_id=DatasetId(dataset_id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )

    load(
        dataset=f"{organization_id}/{dataset_id}",
        bucket="dev-neptune-export-use1",
        db=db,
        use_cache=False,
        smoke_test=False,
    )

    # Models

    patient = db.get_model("patient")
    assert patient == Model(
        name="patient",
        display_name="Patient",
        description="",
        count=2,
        id="0b4b3615-9eaf-425d-9727-bcac29686fd5",
        created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        created_at=iso8601.parse("2019-11-01T20:01:27.027Z"),
        updated_at=iso8601.parse("2019-11-01T20:01:27.027Z"),
        template_id=None,
    )

    assert sorted(db.get_properties(patient), key=lambda p: p.index) == [
        ModelProperty(
            id="7b17c60d-ca2a-4cf5-a4ff-a52bbc32ff17",
            name="name",
            display_name="Name",
            description="",
            index=0,
            locked=False,
            model_title=True,
            required=False,
            data_type=dt.String(),
            default=True,
            default_value=None,
            created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            created_at=iso8601.parse("2019-11-01T20:01:37.633Z"),
            updated_at=iso8601.parse("2019-11-11T15:11:17.383Z"),
        ),
        ModelProperty(
            id="e507b3ef-ade4-4672-83b4-f3f0774fb282",
            name="dob",
            display_name="DOB",
            description="",
            index=1,
            locked=False,
            model_title=False,
            required=False,
            data_type=dt.Date(),
            default=False,
            default_value=None,
            created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            created_at=iso8601.parse("2019-11-11T15:11:17.383Z"),
            updated_at=iso8601.parse("2019-11-11T15:11:17.383Z"),
        ),
    ]

    bicycle = db.get_model("bicycle")
    assert bicycle.name == "bicycle"
    assert bicycle.display_name == "Bicycle"
    assert bicycle.id == "bf858cb5-ae51-4fcf-ad74-b1887946f70f"
    assert bicycle.count == 1
    assert bicycle.template_id == None

    properties = sorted(db.get_properties(bicycle), key=lambda p: p.index)
    assert len(properties) == 2
    brand = properties[0]
    assert brand.name == "brand"

    color = properties[1]
    assert color.name == "color"
    assert color.data_type == dt.Array(
        items=dt.String(), enum=["purple", "blue", "orange", "green", "yellow", "red"]
    )

    # Records
    patients = db.get_all_records("patient")

    alice = Record(
        id=UUID("ecb71447-b684-c589-abda-b673c38edefc"),
        created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        created_at=iso8601.parse("2019-11-01T20:01:58.537Z"),
        updated_at=iso8601.parse("2019-11-11T15:37:02.165Z"),
        values={
            "name": "Alice",
            "dob": neotime.DateTime(year=2004, month=5, day=5, tzinfo=pytz.UTC),
        },
    )
    bob = Record(
        id=UUID("e2b71447-e29d-11c3-24c6-f2ebffd1486a"),
        created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        created_at=iso8601.parse("2019-11-01T20:02:21.113Z"),
        updated_at=iso8601.parse("2019-11-01T20:02:21.113Z"),
        values={
            "name": "Bob",
            # Embedded linked property
            "mother": RecordStub(
                id=UUID("ecb71447-b684-c589-abda-b673c38edefc"), title="Alice"
            ),
        },
    )
    assert sorted(patients.results, key=lambda x: x.values["name"]) == [alice, bob]

    assert db.get_all_records("bicycle").results == [
        Record(
            id=UUID("c8b71de8-cd9c-cc3f-67fe-4e30968d4e50"),
            created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            created_at=iso8601.parse("2019-11-05T13:47:02.841Z"),
            updated_at=iso8601.parse("2019-11-11T15:12:28.042Z"),
            values={"brand": "Bianchi", "color": ["red", "blue"]},
        )
    ]

    # Model relationships
    with db.transaction() as tx:

        assert list(
            db.get_outgoing_model_relationships_tx(tx, patient, one_to_many=True)
        ) == [
            ModelRelationship(
                id="2e754729-684a-4c45-960f-348d68737d4d",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                display_name="Rides",
                description="",
                from_="0b4b3615-9eaf-425d-9727-bcac29686fd5",
                to="bf858cb5-ae51-4fcf-ad74-b1887946f70f",
                one_to_many=True,
                index=None,
                created_at=iso8601.parse("2019-11-05T13:47:17.981Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:17.981Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # This relationship can be created in the Python client with the following:
        #
        #  >>> patient = ds.models()["patient"]
        #  >>> bike = ds.models()["bicycle"]
        #  >>> bob = patient.get_all()[1]
        #  >>> bianchi = bike.get_all()[0]
        #  >>> bianchi.relate_to(bob, relationship_type="belongs_to")
        #
        # This reuses the `belongs_to` name even though that is disallowed through
        # the frontend.  This means that the `belongs_to` CSV contains relationships
        # between proxy packages and records, *and* between records and records.

        assert list(
            db.get_outgoing_model_relationships_tx(tx, bicycle, one_to_many=True)
        ) == [
            ModelRelationship(
                id="175ff55b-b44d-4381-bd59-d4dbc0b9c5f0",
                type="BELONGS_TO",
                name="belongs_to",
                display_name="Belongs To",
                description="",
                from_="bf858cb5-ae51-4fcf-ad74-b1887946f70f",
                to="0b4b3615-9eaf-425d-9727-bcac29686fd5",
                one_to_many=True,
                index=None,
                created_at=iso8601.parse("2019-11-21T16:47:36.918Z"),
                updated_at=iso8601.parse("2019-11-21T16:47:36.918Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Model relationship stubs contain no "to" and "from" models, eg. belongs_to

        assert list(db.get_model_relationship_stubs_tx(tx)) == [
            ModelRelationshipStub(
                id="ccf200d3-e77f-4d9e-bed3-f1f28860152f",
                name="belongs_to",
                display_name="Belongs To",
                description="",
                type="BELONGS_TO",
                created_at=iso8601.parse("2019-11-05T13:44:38.598Z"),
                updated_at=iso8601.parse("2019-11-05T13:44:38.598Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Duplicate @RELATED_TO relationships are created
        assert len(db.get_model_relationships_tx(tx, one_to_many=True)) == 2

        # Record relationships

        assert list(db.get_outgoing_record_relationships_tx(tx, alice)) == [
            RecordRelationship(
                id="d0b71de9-21f9-3557-edda-ad278dd81dc0",
                from_="ecb71447-b684-c589-abda-b673c38edefc",
                to="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                model_relationship_id="2e754729-684a-4c45-960f-348d68737d4d",
                display_name="Rides",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        assert list(
            db.get_outgoing_record_relationships_tx(
                tx, "c8b71de8-cd9c-cc3f-67fe-4e30968d4e50", one_to_many=True
            )
        ) == [
            RecordRelationship(
                id="aeb7476e-55f6-7924-5e43-a83cfa7e4cef",
                from_="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                to="e2b71447-e29d-11c3-24c6-f2ebffd1486a",
                type="BELONGS_TO",
                name="belongs_to",
                model_relationship_id="175ff55b-b44d-4381-bd59-d4dbc0b9c5f0",
                display_name="Belongs To",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                updated_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Linked properties

        assert list(
            db.get_outgoing_model_relationships_tx(tx, patient, one_to_many=False)
        ) == [
            ModelRelationship(
                id="443e141b-f59c-419f-82c1-eed97925b04d",
                type="MOTHER",
                name="mother",
                display_name="Mother",
                description="",
                from_="0b4b3615-9eaf-425d-9727-bcac29686fd5",
                to="0b4b3615-9eaf-425d-9727-bcac29686fd5",
                one_to_many=False,
                index=1,
                created_at=iso8601.parse("2019-11-05T13:43:38.341Z"),
                updated_at=iso8601.parse("2019-11-05T13:43:38.341Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Duplicate @RELATED_TO relationships are created for linked properties
        assert len(db.get_model_relationships_tx(tx, one_to_many=False)) == 1

        assert list(
            db.get_outgoing_record_relationships_tx(tx, bob, one_to_many=False)
        ) == [
            RecordRelationship(
                id="fa3daedd-1761-4730-be7d-bb5de8e1261c",
                from_="e2b71447-e29d-11c3-24c6-f2ebffd1486a",
                to="ecb71447-b684-c589-abda-b673c38edefc",
                type="MOTHER",
                model_relationship_id="443e141b-f59c-419f-82c1-eed97925b04d",
                name="mother",
                display_name="Mother",
                one_to_many=False,
                created_at=iso8601.parse("2019-11-05T13:43:54.116Z"),
                updated_at=iso8601.parse("2019-11-05T13:43:54.116Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

    assert db.get_package_proxies_for_record(alice, limit=10, offset=0) == (
        1,
        [
            PackageProxy(
                id="00b71de7-b42f-1fe9-a83f-824452fe966e",
                proxy_instance_id="460591a0-8079-4979-a860-c3a4b18a32ad",
                package_id=184418,
                package_node_id="N:package:b493794a-1c86-4c18-9fb9-dfdf236b1fe3",
                relationship_type="belongs_to",
                created_at=iso8601.parse("2019-11-05T13:44:38.748Z"),
                updated_at=iso8601.parse("2019-11-05T13:44:38.748Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ],
    )

    # Packages link directly to dataset node
    assert db.count_packages() == 1


@pytest.mark.skip
@pytest.mark.integration
def test_fix_nonexistent_model_relationships(neo4j):
    """
    Case when a record relationship has a schemaRelationshipId that does not
    exist, but a model relationhip (d0b71de9-21f9-3557-edda-ad278dd81dc0) is
    defined that matches the (from)-[relationship]->(to) triple of the
    relationship.

    This corrupted data is from old versions of the graph-ingest tool.
    """

    dataset_id = 30000
    dataset_node_id = "N:dataset:b1154216-d1d7-4484-ad18-81b58fb65484"
    organization_id = 5
    organization_node_id = "N:organization:c905919f-56f5-43ae-9c2a-8d5d542c133b"
    user_id = 114
    user_node_id = "N:user:028058b9-dd8d-4f24-a187-ea56830b379f"

    db = PartitionedDatabase(
        db=neo4j,
        organization_id=OrganizationId(organization_id),
        dataset_id=DatasetId(dataset_id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )

    load(
        dataset=f"{organization_id}/{dataset_id}",
        bucket="dev-neptune-export-use1",
        db=db,
        use_cache=False,
        smoke_test=False,
    )

    with db.transaction() as tx:

        assert list(
            db.get_outgoing_record_relationships_tx(
                tx, "ecb71447-b684-c589-abda-b673c38edefc"
            )
        ) == [
            RecordRelationship(
                id="d0b71de9-21f9-3557-edda-ad278dd81dc0",
                from_="ecb71447-b684-c589-abda-b673c38edefc",
                to="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                model_relationship_id="2e754729-684a-4c45-960f-348d68737d4d",
                display_name="Rides",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        assert list(
            db.get_outgoing_record_relationships_tx(
                tx, "c8b71de8-cd9c-cc3f-67fe-4e30968d4e50", one_to_many=True
            )
        ) == [
            RecordRelationship(
                id="aeb7476e-55f6-7924-5e43-a83cfa7e4cef",
                from_="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                to="e2b71447-e29d-11c3-24c6-f2ebffd1486a",
                type="BELONGS_TO",
                name="belongs_to",
                model_relationship_id="175ff55b-b44d-4381-bd59-d4dbc0b9c5f0",
                display_name="Belongs To",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                updated_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]


@pytest.mark.skip
@pytest.mark.integration
def test_fix_mismatched_model_relationships(neo4j):
    """
    Case when a record relationship has a schemaRelationshipId that does not
    match the (from)-[relationship]->(to) triple of the relationship, but a
    compatible model relationship does exist in the dataset.

    In this case, an extra model relationship (id =
    `4ea7d9e6-b2ae-487c-981c-8202e37343f9`) connects
    (bike)-[rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c]->(bike) and the record
    relationship (person)-[rides]->(bike) incorrectly uses this
    schemaRelationshipId.


    This corrupted data is from old versions of the graph-ingest tool.
    """

    dataset_id = 40000
    dataset_node_id = "N:dataset:b1154216-d1d7-4484-ad18-81b58fb65484"
    organization_id = 5
    organization_node_id = "N:organization:c905919f-56f5-43ae-9c2a-8d5d542c133b"
    user_id = 114
    user_node_id = "N:user:028058b9-dd8d-4f24-a187-ea56830b379f"

    db = PartitionedDatabase(
        db=neo4j,
        organization_id=OrganizationId(organization_id),
        dataset_id=DatasetId(dataset_id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )

    load(
        dataset=f"{organization_id}/{dataset_id}",
        bucket="dev-neptune-export-use1",
        db=db,
        use_cache=False,
        smoke_test=False,
    )

    with db.transaction() as tx:

        assert list(
            db.get_outgoing_record_relationships_tx(
                tx, "ecb71447-b684-c589-abda-b673c38edefc"
            )
        ) == [
            RecordRelationship(
                id="d0b71de9-21f9-3557-edda-ad278dd81dc0",
                from_="ecb71447-b684-c589-abda-b673c38edefc",
                to="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                model_relationship_id="2e754729-684a-4c45-960f-348d68737d4d",
                display_name="Rides",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        assert list(
            db.get_outgoing_record_relationships_tx(
                tx, "c8b71de8-cd9c-cc3f-67fe-4e30968d4e50", one_to_many=True
            )
        ) == [
            RecordRelationship(
                id="aeb7476e-55f6-7924-5e43-a83cfa7e4cef",
                from_="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                to="e2b71447-e29d-11c3-24c6-f2ebffd1486a",
                type="BELONGS_TO",
                name="belongs_to",
                model_relationship_id="175ff55b-b44d-4381-bd59-d4dbc0b9c5f0",
                display_name="Belongs To",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                updated_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]


@pytest.mark.skip
@pytest.mark.integration
def test_fix_multiple_mismatched_model_relationships(neo4j):
    """
    Case when a record relationship has a schemaRelationshipId that does not
    match the (from)-[relationship]->(to) triple of the relationship, but a
    compatible model relationship does exist in the dataset.

    In this case, an extra model relationship (id =
    `4ea7d9e6-b2ae-487c-981c-8202e37343f9`) connects
    (bike)-[rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c]->(bike) and the record
    relationship (person)-[rides]->(bike) incorrectly uses this
    schemaRelationshipId.

    AND

    Another record relationship, with the same relationship type, also claiming
    the same schemaRelationshipId, but actually connecting different models.

    This corrupted data is from old versions of the graph-ingest tool.
    """
    dataset_id = 50000
    dataset_node_id = "N:dataset:b1154216-d1d7-4484-ad18-81b58fb65484"
    organization_id = 5
    organization_node_id = "N:organization:c905919f-56f5-43ae-9c2a-8d5d542c133b"
    user_id = 114
    user_node_id = "N:user:028058b9-dd8d-4f24-a187-ea56830b379f"

    db = PartitionedDatabase(
        db=neo4j,
        organization_id=OrganizationId(organization_id),
        dataset_id=DatasetId(dataset_id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )

    load(
        dataset=f"{organization_id}/{dataset_id}",
        bucket="dev-neptune-export-use1",
        db=db,
        use_cache=False,
        smoke_test=False,
    )

    with db.transaction() as tx:

        assert list(
            db.get_outgoing_record_relationships_tx(
                tx, "ecb71447-b684-c589-abda-b673c38edefc"
            )
        ) == [
            RecordRelationship(
                id="d0b71de9-21f9-3557-edda-ad278dd81dc0",
                from_="ecb71447-b684-c589-abda-b673c38edefc",
                to="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                model_relationship_id="2e754729-684a-4c45-960f-348d68737d4d",
                display_name="Rides",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        assert sorted(
            db.get_outgoing_record_relationships_tx(
                tx, "c8b71de8-cd9c-cc3f-67fe-4e30968d4e50", one_to_many=True
            ),
            key=lambda r: r.type,
        ) == [
            RecordRelationship(
                id="aeb7476e-55f6-7924-5e43-a83cfa7e4cef",
                from_="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                to="e2b71447-e29d-11c3-24c6-f2ebffd1486a",
                type="BELONGS_TO",
                name="belongs_to",
                model_relationship_id="175ff55b-b44d-4381-bd59-d4dbc0b9c5f0",
                display_name="Belongs To",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                updated_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            ),
            RecordRelationship(
                id="83227fcd-ec4d-47e6-a3a9-4c9fbba4a2f3",
                from_="c8b71de8-cd9c-cc3f-67fe-4e30968d4e50",
                to="e2b71447-e29d-11c3-24c6-f2ebffd1486a",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                model_relationship_id="92b6a32f-0597-4a7a-944b-27b39998283e",
                display_name="Rides",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            ),
        ]


@pytest.mark.skip
@pytest.mark.integration
def test_rewrite_ids_and_import(neo4j):
    """
    Test that UUIDs are remapped to the exact correct place with a manually
    defined remapping.
    """

    dataset_id = 60000
    dataset_node_id = "N:dataset:b1154216-d1d7-4484-ad18-81b58fb65484"
    organization_id = 5
    organization_node_id = "N:organization:c905919f-56f5-43ae-9c2a-8d5d542c133b"
    user_id = 114
    user_node_id = "N:user:028058b9-dd8d-4f24-a187-ea56830b379f"

    db = PartitionedDatabase(
        db=neo4j,
        organization_id=OrganizationId(organization_id),
        dataset_id=DatasetId(dataset_id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )

    REMAPPING = {
        "0b4b3615-9eaf-425d-9727-bcac29686fd5": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "7b17c60d-ca2a-4cf5-a4ff-a52bbc32ff17": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "e507b3ef-ade4-4672-83b4-f3f0774fb282": "cccccccc-cccc-cccc-cccc-cccccccccccc",
        "bf858cb5-ae51-4fcf-ad74-b1887946f70f": "dddddddd-dddd-dddd-dddd-dddddddddddd",
        "a99b09f5-caa6-4282-aa0e-cf56bde89254": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        "42fa4eb9-51cc-4c59-b550-ac24d6d5024a": "ffffffff-ffff-ffff-ffff-ffffffffffff",
        "ecb71447-b684-c589-abda-b673c38edefc": "00000000-0000-0000-0000-000000000000",
        "e2b71447-e29d-11c3-24c6-f2ebffd1486a": "11111111-1111-1111-1111-111111111111",
        "c8b71de8-cd9c-cc3f-67fe-4e30968d4e50": "22222222-2222-2222-2222-222222222222",
        "2e754729-684a-4c45-960f-348d68737d4d": "33333333-3333-3333-3333-333333333333",
        "175ff55b-b44d-4381-bd59-d4dbc0b9c5f0": "44444444-4444-4444-4444-444444444444",
        "ccf200d3-e77f-4d9e-bed3-f1f28860152f": "55555555-5555-5555-5555-555555555555",
        "443e141b-f59c-419f-82c1-eed97925b04d": "66666666-6666-6666-6666-666666666666",
        "d0b71de9-21f9-3557-edda-ad278dd81dc0": "77777777-7777-7777-7777-777777777777",
        "aeb7476e-55f6-7924-5e43-a83cfa7e4cef": "88888888-8888-8888-8888-888888888888",
        "fa3daedd-1761-4730-be7d-bb5de8e1261c": "99999999-9999-9999-9999-999999999999",
        "00b71de7-b42f-1fe9-a83f-824452fe966e": "aaaaaaaa-aaaa-aaaa-aaaa-bbbbbbbbbbbb",
        "460591a0-8079-4979-a860-c3a4b18a32ad": "aaaaaaaa-aaaa-aaaa-aaaa-cccccccccccc",
    }

    def generate_new_id(old_id):
        new_id = REMAPPING.get(old_id, None)
        if new_id is None:
            return old_id
        return new_id

    load(
        dataset=f"{organization_id}/{dataset_id}",
        bucket="dev-neptune-export-use1",
        db=db,
        use_cache=False,
        smoke_test=False,
        remap_ids=True,
        generate_new_id=generate_new_id,
    )

    # Models

    patient = db.get_model("patient")
    assert patient == Model(
        name="patient",
        display_name="Patient",
        description="",
        count=2,
        id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        created_at=iso8601.parse("2019-11-01T20:01:27.027Z"),
        updated_at=iso8601.parse("2019-11-01T20:01:27.027Z"),
        template_id=None,
    )

    assert sorted(db.get_properties(patient), key=lambda p: p.index) == [
        ModelProperty(
            id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            name="name",
            display_name="Name",
            description="",
            index=0,
            locked=False,
            model_title=True,
            required=False,
            data_type=dt.String(),
            default=True,
            default_value=None,
            created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            created_at=iso8601.parse("2019-11-01T20:01:37.633Z"),
            updated_at=iso8601.parse("2019-11-11T15:11:17.383Z"),
        ),
        ModelProperty(
            id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            name="dob",
            display_name="DOB",
            description="",
            index=1,
            locked=False,
            model_title=False,
            required=False,
            data_type=dt.Date(),
            default=False,
            default_value=None,
            created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            created_at=iso8601.parse("2019-11-11T15:11:17.383Z"),
            updated_at=iso8601.parse("2019-11-11T15:11:17.383Z"),
        ),
    ]

    bicycle = db.get_model("bicycle")
    assert bicycle.name == "bicycle"
    assert bicycle.display_name == "Bicycle"
    assert bicycle.id == "dddddddd-dddd-dddd-dddd-dddddddddddd"
    assert bicycle.count == 1
    assert bicycle.template_id == None

    properties = sorted(db.get_properties(bicycle), key=lambda p: p.index)
    assert len(properties) == 2
    brand = properties[0]
    assert brand.name == "brand"
    assert brand.id == "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"

    color = properties[1]
    assert color.name == "color"
    assert color.data_type == dt.Array(
        items=dt.String(), enum=["purple", "blue", "orange", "green", "yellow", "red"]
    )
    assert color.id == "ffffffff-ffff-ffff-ffff-ffffffffffff"

    # Records
    patients = db.get_all_records("patient")

    alice = Record(
        id=UUID("00000000-0000-0000-0000-000000000000"),
        created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        created_at=iso8601.parse("2019-11-01T20:01:58.537Z"),
        updated_at=iso8601.parse("2019-11-11T15:37:02.165Z"),
        values={
            "name": "Alice",
            "dob": neotime.DateTime(year=2004, month=5, day=5, tzinfo=pytz.UTC),
        },
    )
    bob = Record(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
        created_at=iso8601.parse("2019-11-01T20:02:21.113Z"),
        updated_at=iso8601.parse("2019-11-01T20:02:21.113Z"),
        values={
            "name": "Bob",
            # Embedded linked property
            "mother": RecordStub(
                id=UUID("00000000-0000-0000-0000-000000000000"), title="Alice"
            ),
        },
    )
    assert sorted(patients.results, key=lambda x: x.values["name"]) == [alice, bob]

    assert db.get_all_records("bicycle").results == [
        Record(
            id=UUID("22222222-2222-2222-2222-222222222222"),
            created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            created_at=iso8601.parse("2019-11-05T13:47:02.841Z"),
            updated_at=iso8601.parse("2019-11-11T15:12:28.042Z"),
            values={"brand": "Bianchi", "color": ["red", "blue"]},
        )
    ]

    # Model relationships
    with db.transaction() as tx:
        assert list(
            db.get_outgoing_model_relationships_tx(tx, patient, one_to_many=True)
        ) == [
            ModelRelationship(
                id="33333333-3333-3333-3333-333333333333",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                display_name="Rides",
                description="",
                from_="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                to="dddddddd-dddd-dddd-dddd-dddddddddddd",
                one_to_many=True,
                index=None,
                created_at=iso8601.parse("2019-11-05T13:47:17.981Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:17.981Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        assert list(
            db.get_outgoing_model_relationships_tx(tx, bicycle, one_to_many=True)
        ) == [
            ModelRelationship(
                id="44444444-4444-4444-4444-444444444444",
                type="BELONGS_TO",
                name="belongs_to",
                display_name="Belongs To",
                description="",
                from_="dddddddd-dddd-dddd-dddd-dddddddddddd",
                to="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                one_to_many=True,
                index=None,
                created_at=iso8601.parse("2019-11-21T16:47:36.918Z"),
                updated_at=iso8601.parse("2019-11-21T16:47:36.918Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Model relationship stubs contain no "to" and "from" models, eg. belongs_to

        assert list(db.get_model_relationship_stubs_tx(tx)) == [
            ModelRelationshipStub(
                id="55555555-5555-5555-5555-555555555555",
                name="belongs_to",
                display_name="Belongs To",
                description="",
                type="BELONGS_TO",
                created_at=iso8601.parse("2019-11-05T13:44:38.598Z"),
                updated_at=iso8601.parse("2019-11-05T13:44:38.598Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Duplicate @RELATED_TO relationships are created
        assert len(db.get_model_relationships_tx(tx, one_to_many=True)) == 2

        # Record relationships

        assert list(db.get_outgoing_record_relationships_tx(tx, alice)) == [
            RecordRelationship(
                id="77777777-7777-7777-7777-777777777777",
                from_="00000000-0000-0000-0000-000000000000",
                to="22222222-2222-2222-2222-222222222222",
                type="RIDES",
                name="rides_c83d5af0-ffd2-11e9-b8f0-1b1d6297ff8c",
                model_relationship_id="33333333-3333-3333-3333-333333333333",
                display_name="Rides",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                updated_at=iso8601.parse("2019-11-05T13:47:46.032Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        assert list(
            db.get_outgoing_record_relationships_tx(
                tx, "22222222-2222-2222-2222-222222222222", one_to_many=True
            )
        ) == [
            RecordRelationship(
                id="88888888-8888-8888-8888-888888888888",
                from_="22222222-2222-2222-2222-222222222222",
                to="11111111-1111-1111-1111-111111111111",
                type="BELONGS_TO",
                name="belongs_to",
                model_relationship_id="44444444-4444-4444-4444-444444444444",
                display_name="Belongs To",
                one_to_many=True,
                created_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                updated_at=iso8601.parse("2019-11-21T16:47:36.938Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Linked properties

        assert list(
            db.get_outgoing_model_relationships_tx(tx, patient, one_to_many=False)
        ) == [
            ModelRelationship(
                id="66666666-6666-6666-6666-666666666666",
                type="MOTHER",
                name="mother",
                display_name="Mother",
                description="",
                from_="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                to="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                one_to_many=False,
                index=1,
                created_at=iso8601.parse("2019-11-05T13:43:38.341Z"),
                updated_at=iso8601.parse("2019-11-05T13:43:38.341Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

        # Duplicate @RELATED_TO relationships are created for linked properties
        assert len(db.get_model_relationships_tx(tx, one_to_many=False)) == 1

        assert list(
            db.get_outgoing_record_relationships_tx(tx, bob, one_to_many=False)
        ) == [
            RecordRelationship(
                id="99999999-9999-9999-9999-999999999999",
                from_="11111111-1111-1111-1111-111111111111",
                to="00000000-0000-0000-0000-000000000000",
                type="MOTHER",
                model_relationship_id="66666666-6666-6666-6666-666666666666",
                name="mother",
                display_name="Mother",
                one_to_many=False,
                created_at=iso8601.parse("2019-11-05T13:43:54.116Z"),
                updated_at=iso8601.parse("2019-11-05T13:43:54.116Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ]

    assert db.get_package_proxies_for_record(alice, limit=10, offset=0) == (
        1,
        [
            PackageProxy(
                id="aaaaaaaa-aaaa-aaaa-aaaa-bbbbbbbbbbbb",
                proxy_instance_id="aaaaaaaa-aaaa-aaaa-aaaa-cccccccccccc",
                package_id=184418,
                package_node_id="N:package:b493794a-1c86-4c18-9fb9-dfdf236b1fe3",
                relationship_type="belongs_to",
                created_at=iso8601.parse("2019-11-05T13:44:38.748Z"),
                updated_at=iso8601.parse("2019-11-05T13:44:38.748Z"),
                created_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
                updated_by="N:user:06080380-fb56-46eb-8c70-f24112aff878",
            )
        ],
    )

    # Packages link directly to dataset node
    assert db.count_packages() == 1


@pytest.mark.skip
@pytest.mark.integration
def test_rewrite_ids_randomly_and_import(neo4j):
    """
    Test that UUIDs are remapped using the default random remapper.
    """
    dataset_id = 70000
    dataset_node_id = "N:dataset:b1154216-d1d7-4484-ad18-81b58fb65484"
    organization_id = 5
    organization_node_id = "N:organization:c905919f-56f5-43ae-9c2a-8d5d542c133b"
    user_id = 114
    user_node_id = "N:user:028058b9-dd8d-4f24-a187-ea56830b379f"

    db = PartitionedDatabase(
        db=neo4j,
        organization_id=OrganizationId(organization_id),
        dataset_id=DatasetId(dataset_id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )

    load(
        dataset=f"{organization_id}/{dataset_id}",
        bucket="dev-neptune-export-use1",
        db=db,
        use_cache=False,
        smoke_test=False,
        remap_ids=True,
    )

    # Models

    patient = db.get_model("patient")
    assert patient is not None
    assert len(db.get_properties(patient)) == 2
    assert patient.id != UUID("0b4b3615-9eaf-425d-9727-bcac29686fd5")

    bicycle = db.get_model("bicycle")
    assert bicycle is not None
    assert len(db.get_properties(bicycle)) == 2
    assert bicycle.id != UUID("bf858cb5-ae51-4fcf-ad74-b1887946f70f")

    # Records
    patients = db.get_all_records("patient")

    alice = [r for r in patients if r.values["name"] == "Alice"][0]
    assert alice.id != "ecb71447-b684-c589-abda-b673c38edefc"
    bob = [r for r in patients if r.values["name"] == "Bob"][0]
    assert bob.id != UUID("e2b71447-e29d-11c3-24c6-f2ebffd1486a")

    assert len(db.get_all_records("bicycle").results) == 1
    bianchi = db.get_all_records("bicycle").results[0]
    assert bianchi.id != UUID("c8b71de8-cd9c-cc3f-67fe-4e30968d4e50")

    # Model relationships
    with db.transaction() as tx:
        assert (
            len(
                list(
                    db.get_outgoing_model_relationships_tx(
                        tx, patient, one_to_many=True
                    )
                )
            )
            == 1
        )

        assert (
            len(
                list(
                    db.get_outgoing_model_relationships_tx(
                        tx, bicycle, one_to_many=True
                    )
                )
            )
            == 1
        )

        # Model relationship stubs contain no "to" and "from" models, eg. belongs_to

        assert len(list(db.get_model_relationship_stubs_tx(tx))) == 1

        # Duplicate @RELATED_TO relationships are created
        assert len(list(db.get_model_relationships_tx(tx, one_to_many=True))) == 2

        # Record relationships

        assert len(list(db.get_outgoing_record_relationships_tx(tx, alice))) == 1

        assert (
            len(
                list(
                    db.get_outgoing_record_relationships_tx(
                        tx, bianchi, one_to_many=True
                    )
                )
            )
            == 1
        )

        # Linked properties

        assert (
            len(
                list(
                    db.get_outgoing_model_relationships_tx(
                        tx, patient, one_to_many=False
                    )
                )
            )
            == 1
        )

        # Duplicate @RELATED_TO relationships are created for linked properties
        assert len(list(db.get_model_relationships_tx(tx, one_to_many=False))) == 1

        assert (
            len(
                list(
                    db.get_outgoing_record_relationships_tx(tx, bob, one_to_many=False)
                )
            )
            == 1
        )

    assert len(db.get_package_proxies_for_record(alice, limit=10, offset=0)[1]) == 1

    # Packages link directly to dataset node
    assert db.count_packages() == 1
