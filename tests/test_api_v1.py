import copy
import json
import logging
import time
import uuid
from itertools import chain
from uuid import UUID

import pytest
from more_itertools import unique_justseen

from core.clients import (
    CreateModel,
    CreateModelProperty,
    CreateRecord,
    DeleteModel,
    DeleteModelProperty,
    DeleteRecord,
    UpdateModel,
    UpdateModelProperty,
    UpdateRecord,
)
from server.api.v1 import common
from server.db import labels
from server.errors import CannotSortRecordsError
from server.models import (
    ORDER_BY_CREATED_AT_FIELDS,
    ORDER_BY_UPDATED_AT_FIELDS,
    OrderByField,
)
from server.models import datatypes as dt
from server.models import legacy

log = logging.getLogger(__file__)


def drop_volatile(data):
    """
    Remove fields that change between test runs from the dictionary `data`.
    """
    clean = copy.copy(data)

    for field in ["id", "createdAt", "updatedAt", "createdBy", "updatedBy"]:
        clean.pop(field, None)

    return clean


def get_property(data, key):
    """
    Get a property, either from a JSON response dictionary or a dataclass.
    """
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


@pytest.fixture(scope="function")
def concept(create_concept, name="patient", display_name=None, description=None):
    return create_concept(name, display_name, description)


@pytest.fixture(scope="function")
def create_concept(client, auth_headers, trace_id_headers, valid_dataset):
    dataset_id, _ = valid_dataset

    def build(name="patient", display_name=None, description=None):
        if display_name is None:
            display_name = name.capitalize()
        if description is None:
            description = f"{name.capitalize()} model"
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/concepts",
            headers=dict(**auth_headers, **trace_id_headers),
            json={
                "name": name,
                "displayName": display_name,
                "description": description,
            },
        )
        if r.status_code > 299:
            log.warning(f"model: {r.json}")
        assert r.status_code == 201
        return r.json

    return build


@pytest.fixture(scope="function")
def get_concept(client, auth_headers, trace_id_headers, valid_dataset):
    dataset_id, _ = valid_dataset

    def get(id_or_name):
        r = client.get(
            f"/v1/datasets/{dataset_id.id}/concepts/{id_or_name}",
            headers=dict(**auth_headers, **trace_id_headers),
        )
        if r.status_code > 299:
            log.warning(f"create_record: {r.json}")
        assert r.status_code == 200
        return r.json

    return get


@pytest.fixture(scope="function")
def create_concept_instance(client, auth_headers, trace_id_headers, valid_dataset):
    dataset_id, _ = valid_dataset

    def build(concept, values):
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/concepts/{concept['id']}/instances",
            headers=dict(**auth_headers, **trace_id_headers),
            json={"values": values},
        )
        if r.status_code > 299:
            log.warning(f"create_concept_instance: {r.json}")
        assert r.status_code == 201
        return r.json

    return build


@pytest.fixture(scope="function")
def create_concept_instance_batch(
    client, auth_headers, trace_id_headers, valid_dataset
):
    dataset_id, _ = valid_dataset

    def build(concept, values):
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/concepts/{get_property(concept, 'id')}/instances/batch",
            headers=dict(**auth_headers, **trace_id_headers),
            json=[{"values": v} for v in values],
        )
        if r.status_code > 299:
            log.warning(f"create_concept_instance_batch: {r.json}")
        assert r.status_code == 200
        return r.json

    return build


@pytest.fixture(scope="function")
def create_property(client, auth_headers, trace_id_headers, valid_dataset):
    dataset_id, _ = valid_dataset

    def build(concept_id, name="name", data_type=dt.String(), title=True):
        r = client.put(
            f"/v1/datasets/{dataset_id.id}/concepts/{concept_id}/properties",
            headers=dict(**auth_headers, **trace_id_headers),
            json=[
                {
                    "name": name,
                    "displayName": name,
                    "description": "",
                    "dataType": dt.serialize(data_type),
                    "locked": False,
                    "conceptTitle": title,
                    "required": False,
                    "default": True,
                }
            ],
        )
        if r.status_code > 299:
            log.warning(f"create_property: {r.json}")
        assert r.status_code == 200
        return r.json

    return build


@pytest.fixture(scope="function")
def create_linked_property(client, auth_headers, trace_id_headers, valid_dataset):
    dataset_id, _ = valid_dataset

    def build(concept, name, to, display_name=None):
        if display_name is None:
            display_name = name

        r = client.post(
            f"/v1/datasets/{dataset_id.id}/concepts/{get_property(concept, 'id')}/linked",
            headers=dict(**auth_headers, **trace_id_headers),
            json={
                "name": name,
                "displayName": display_name,
                "to": get_property(to, "id"),
            },
        )
        if r.status_code > 299:
            log.warning(f"create_linked_property: {r.json}")
        assert r.status_code == 201
        return r.json

    return build


@pytest.fixture(scope="function")
def create_linked_property_instance(
    client, auth_headers, trace_id_headers, valid_dataset
):
    dataset_id, _ = valid_dataset

    def build(concept, concept_instance, linked_property, to_concept_instance):
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/concepts/{get_property(concept, 'id')}/instances/{get_property(concept_instance, 'id')}/linked",
            headers=dict(**auth_headers, **trace_id_headers),
            json={
                "name": get_property(linked_property, "name"),
                "schemaLinkedPropertyId": get_property(linked_property, "id"),
                "to": get_property(to_concept_instance, "id"),
            },
        )
        if r.status_code > 299:
            log.warning(f"create_linked_property: {r.json}")
        assert r.status_code == 201
        return r.json

    return build


@pytest.fixture(scope="function")
def create_concept_relationship(client, auth_headers, trace_id_headers, valid_dataset):
    dataset_id, _ = valid_dataset

    def build(name, displayName, from_, to):
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/relationships",
            headers=dict(**auth_headers, **trace_id_headers),
            json={
                "name": name,
                "displayName": displayName,
                "description": displayName,
                "schema": [],
                "from": from_,
                "to": to,
            },
        )
        if r.status_code > 299:
            log.warning(f"create_concept_relationship: {r.json}")
        assert r.status_code == 201
        return r.json

    return build


@pytest.fixture(scope="function")
def create_concept_relationship_stub(
    client, auth_headers, trace_id_headers, valid_dataset
):
    dataset_id, _ = valid_dataset

    def build(name, displayName):
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/relationships",
            headers=dict(**auth_headers, **trace_id_headers),
            json={
                "name": name,
                "displayName": displayName,
                "description": displayName,
                "schema": [],
            },
        )
        if r.status_code > 299:
            log.warning(f"create_concept_relationship: {r.json}")
        assert r.status_code == 201
        return r.json

    return build


@pytest.fixture(scope="function")
def create_concept_instance_relationship(
    client, auth_headers, trace_id_headers, valid_dataset
):
    dataset_id, _ = valid_dataset

    def build(from_, type_, to):
        # Creation fails for mismatched relationship type:
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/relationships/{type_}/instances",
            headers=dict(**auth_headers, **trace_id_headers),
            json={"from": from_, "to": to, "values": []},
        )
        if r.status_code > 299:
            log.warning(f"create_concept_instance_relationship: {r.json}")
        assert r.status_code == 200
        return r.json

    return build


@pytest.fixture(scope="function")
def create_concept_instance_relationship_batch(
    client, auth_headers, trace_id_headers, valid_dataset
):
    dataset_id, _ = valid_dataset

    def make_relationship_json(source, destination):
        return {
            "values": [],
            "to": get_property(destination, "id"),
            "from": get_property(source, "id"),
        }

    def build(from_, type_, to):
        # Creation fails for mismatched relationship type:
        assert len(from_) == len(to)
        values = [
            make_relationship_json(source, destination)
            for source, destination in zip(from_, to)
        ]
        r = client.post(
            f"/v1/datasets/{dataset_id.id}/relationships/{type_}/instances/batch",
            headers=dict(**auth_headers, **trace_id_headers),
            json=values,
        )
        if r.status_code > 299:
            log.warning(f"create_concept_instance_relationship: {r.json}")
        assert r.status_code == 200
        return r

    return build


def test_concepts(
    get,
    post,
    put,
    delete,
    trace_id,
    valid_organization,
    valid_dataset,
    valid_user,
    jobs_client,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    r = get("/v1/datasets/{dataset}/concepts")

    assert r.status_code == 200
    assert r.json == []

    r = post(
        "/v1/datasets/{dataset}/concepts",
        json={"name": "patient", "displayName": "Patient", "description": "Test model"},
    )
    assert r.status_code == 201
    created = r.json
    assert created["createdBy"] == user_node_id
    assert created["updatedBy"] == user_node_id
    assert created["count"] == 0

    # "CreateModel" should be emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateModel(id=UUID(created["id"]), name=created["name"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    r = get("/v1/datasets/{dataset}/concepts")
    assert r.status_code == 200
    assert r.json == [created]

    model_id = created["id"]
    r = get("/v1/datasets/{dataset}/concepts/{model_id}", model_id=model_id)
    assert r.status_code == 200
    assert r.json == created

    r = put(
        "/v1/datasets/{dataset}/concepts/{model_id}",
        model_id=model_id,
        json={"name": "doctor", "displayName": "Doctor", "description": "Test model 2"},
    )
    assert r.status_code == 200
    updated = r.json
    assert updated["name"] == "doctor"
    assert updated["displayName"] == "Doctor"
    assert updated["description"] == "Test model 2"
    assert updated["count"] == 0

    # "UpdateModel" should be emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=UpdateModel(id=UUID(updated["id"]), name=updated["name"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    r = delete("/v1/datasets/{dataset}/concepts/{model_id}", model_id=model_id)
    assert r.status_code == 200
    deleted = r.json
    assert deleted["name"] == "doctor"
    assert deleted["displayName"] == "Doctor"
    assert deleted["description"] == "Test model 2"

    # "DeleteModel" should be emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=DeleteModel(id=UUID(deleted["id"]), name=deleted["name"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    r = get("/v1/datasets/{dataset}/concepts/{model_id}", model_id=model_id)
    assert r.status_code == 404


def test_concepts_with_template_ids(get, post, put, delete, valid_user):
    _, user_node_id = valid_user

    template_id_1 = uuid.uuid4()
    r = post(
        "/v1/datasets/{dataset}/concepts",
        json={
            "name": "patient",
            "displayName": "Patient",
            "description": "Test model",
            "templateId": template_id_1,
        },
    )
    assert r.status_code == 201
    assert r.json["templateId"] == str(template_id_1)

    model_id = r.json["id"]

    template_id_2 = uuid.uuid4()
    r = put(
        "/v1/datasets/{dataset}/concepts/{model_id}",
        json={
            "name": "patient",
            "displayName": "Patient",
            "description": "Test model",
            "templateId": template_id_2,
        },
        model_id=model_id,
    )
    assert r.status_code == 200
    assert r.json["templateId"] == str(template_id_2)

    r = get("/v1/datasets/{dataset}/concepts/{model_id}", model_id=model_id)
    assert r.status_code == 200
    assert r.json["templateId"] == str(template_id_2)

    r = put(
        "/v1/datasets/{dataset}/concepts/{model_id}",
        json={"name": "patient", "displayName": "Patient", "description": "Test model"},
        model_id=model_id,
    )
    assert r.status_code == 200
    assert r.json["templateId"] == None


def test_concept_properties(
    get,
    post,
    put,
    delete,
    concept,
    trace_id,
    jobs_client,
    valid_organization,
    valid_dataset,
    valid_user,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    # Create a property
    r = put(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties",
        concept_id=concept["id"],
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "All 'bout it",
                "dataType": "String",
                "locked": False,
                "conceptTitle": True,
                "required": False,
                "default": False,
            },
            {
                "name": "age",
                "displayName": "Age",
                "description": "",
                "dataType": "String",
                "locked": False,
                "conceptTitle": False,
                "required": False,
                "default": True,
            },
        ],
    )
    assert r.status_code == 200
    name_id, age_id = [p["id"] for p in r.json]

    # "UpdateModelProperty" should be emitted:
    jobs_client.send_changelog_events.assert_called_with(
        dataset_id=dataset_id.id,
        events=[
            CreateModelProperty(
                property_name="name",
                model_id=UUID(concept["id"]),
                model_name=concept["name"],
            ),
            CreateModelProperty(
                property_name="age",
                model_id=UUID(concept["id"]),
                model_name=concept["name"],
            ),
        ],
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    r = get("/v1/datasets/{dataset}/concepts/{concept_id}", concept_id=concept["id"])
    assert r.status_code == 200
    assert r.json["propertyCount"] == 2

    # Can get the property
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    assert [drop_volatile(p) for p in r.json] == [
        {
            "name": "name",
            "displayName": "Name",
            "description": "All 'bout it",
            "dataType": "String",  # Returns simple data types
            "locked": False,
            "conceptTitle": True,
            "required": False,
            "default": False,
            "defaultValue": None,
            "index": 0,
        },
        {
            "name": "age",
            "displayName": "Age",
            "description": "",
            "dataType": "String",
            "locked": False,
            "conceptTitle": False,
            "required": False,
            "default": True,
            "defaultValue": None,
            "index": 1,
        },
    ]

    # Delete the property
    r = delete(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties/{age_id}",
        concept_id=concept["id"],
        age_id=age_id,
    )
    assert r.status_code == 200

    # "UpdateModel" should be emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=DeleteModelProperty(
            model_id=UUID(concept["id"]),
            model_name=concept["name"],
            property_name="age",
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # The property no longer exists
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    assert [p["id"] for p in r.json] == [name_id]

    r = get("/v1/datasets/{dataset}/concepts/{concept_id}", concept_id=concept["id"])
    assert r.status_code == 200
    assert r.json["propertyCount"] == 1


def test_concept_properties_have_nullable_fields(
    get, post, put, delete, concept, valid_user
):
    _, user_node_id = valid_user

    # Create a property with a null required field
    create_properties_response = put(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties",
        concept_id=concept["id"],
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "All 'bout it",
                "dataType": "String",
                "locked": False,
                "conceptTitle": True,
                "required": None,  # This is passed in as an explicit "None"
                "default": False,
            }
        ],
    )
    assert create_properties_response.status_code == 200

    get_concept_response = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}", concept_id=concept["id"]
    )
    assert get_concept_response.status_code == 200
    assert get_concept_response.json["propertyCount"] == 1

    # Get the property
    get_properties_response = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties",
        concept_id=concept["id"],
    )
    assert get_properties_response.status_code == 200
    assert [drop_volatile(p) for p in get_properties_response.json] == [
        {
            "name": "name",
            "displayName": "Name",
            "description": "All 'bout it",
            "dataType": "String",
            "locked": False,
            "conceptTitle": True,
            "required": False,  # This should have defaulted to false
            "default": False,
            "defaultValue": None,
            "index": 0,
        }
    ]


def test_concept_instances(
    audit_logger,
    configure_get,
    configure_post,
    put,
    delete,
    concept,
    create_property,
    trace_id,
    jobs_client,
    valid_organization,
    valid_dataset,
    valid_user,
):
    get = configure_get(audit_logger)
    post = configure_post(audit_logger)

    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    # Create a property
    r = put(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties",
        concept_id=concept["id"],
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "locked": False,
                "conceptTitle": True,
                "required": True,
                "default": False,
            },
            {
                "name": "age",
                "displayName": "Age",
                "description": "",
                "dataType": "Long",
                "locked": False,
                "conceptTitle": False,
                "required": False,
                "default": True,
            },
            {
                "name": "occupation",
                "displayName": "Occupation",
                "description": "",
                "dataType": "String",
                "locked": False,
                "conceptTitle": False,
                "required": False,
                "default": True,
            },
        ],
    )
    assert r.status_code == 200

    # Try to create a new record with a missing "value" for a non-required
    # property:
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
        json={
            "values": [
                {"name": "occupation", "value": "Student"},
                {"name": "name"},
                {"name": "age", "value": "25"},
            ]
        },
    )
    assert r.status_code == 400

    # Missing "value" properties should work for non-required properties.
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
        json={
            "values": [
                {"name": "occupation"},
                {"name": "name", "value": "Homer"},
                {"name": "age", "value": "35"},
            ]
        },
    )
    assert r.status_code == 201
    record = r.json
    homer = record
    assert record["type"] == "patient"
    assert record["createdBy"] == user_node_id
    assert record["updatedBy"] == user_node_id
    values = record["values"]
    assert len(values) == 3
    assert values[0]["name"] == "name"
    assert values[0]["value"] == "Homer"
    assert values[1]["name"] == "age"
    assert values[1]["value"] == 35
    assert values[2]["name"] == "occupation"
    assert values[2]["value"] is None

    # Create a new record -- all values present
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
        json={
            "values": [
                {"name": "occupation", "value": "Student"},
                {"name": "name", "value": "Alice"},
                {"name": "age", "value": "25"},
            ]
        },
    )
    assert r.status_code == 201
    record = r.json
    alice = record
    assert record["type"] == "patient"
    assert record["createdBy"] == user_node_id
    assert record["updatedBy"] == user_node_id
    values = record["values"]
    assert len(values) == 3
    assert values[0]["name"] == "name"
    assert values[0]["value"] == "Alice"
    assert values[1]["name"] == "age"
    assert values[1]["value"] == 25
    assert values[2]["name"] == "occupation"
    assert values[2]["value"] == "Student"

    # Make sure "CreateRecord" was emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(id=UUID(record["id"]), name="Alice", model_id=concept["id"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Get the record
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{record_id}",
        concept_id=concept["id"],
        record_id=record["id"],
    )

    assert r.status_code == 200
    assert r.json == record
    returned_record = r.json
    assert returned_record["type"] == "patient"
    assert returned_record["createdBy"] == user_node_id
    assert returned_record["updatedBy"] == user_node_id
    returned_values = returned_record["values"]
    assert len(returned_values) == 3
    assert returned_values[0]["name"] == "name"
    assert returned_values[0]["value"] == "Alice"
    assert returned_values[1]["name"] == "age"
    assert returned_values[1]["value"] == 25
    assert returned_values[2]["name"] == "occupation"
    assert returned_values[2]["value"] == "Student"

    # Get all records
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == homer
    assert all_records[1] == alice

    # Get all records, sorted by age ascending (by default)
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=age",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == alice
    assert all_records[1] == homer

    # Get all records, sorted by age descending
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=age&ascending=false",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == homer
    assert all_records[1] == alice

    # Get all records, sorted by nullable occupation ascending
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=occupation&ascending=true",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == alice
    assert all_records[1] == homer

    # Get all records, sorted by nullable occupation descending
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=occupation&ascending=false",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == homer
    assert all_records[1] == alice

    # Get all records, sorted by a property that does not exist
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=foobar",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == homer
    assert all_records[1] == alice

    # Get all records, sorted by a property that does not exist ascending
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=foobar&ascending=true",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == homer
    assert all_records[1] == alice

    # Get all records, sorted by createdAt ascending (by default)
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=createdAt",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == homer
    assert all_records[1] == alice

    # Get all records, sorted by createdAt descending
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances?orderBy=createdAt&ascending=false",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    all_records = r.json
    assert len(all_records) == 2
    assert all_records[0] == alice
    assert all_records[1] == homer

    # Update the values of the record
    r = put(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{record_id}",
        concept_id=concept["id"],
        record_id=record["id"],
        json={
            "values": [
                {"name": "occupation", "value": "Teacher"},
                {"name": "name", "value": "Bob"},
                {"name": "age", "value": "39"},
            ]
        },
    )
    assert r.status_code == 200
    record = r.json
    values = record["values"]
    assert len(values) == 3
    assert values[0]["name"] == "name"
    assert values[0]["value"] == "Bob"
    assert values[1]["name"] == "age"
    assert values[1]["value"] == 39
    assert values[2]["name"] == "occupation"
    assert values[2]["value"] == "Teacher"

    # Emit UpdateRecord
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=UpdateRecord(
            id=UUID(record["id"]),
            name="Alice",
            model_id=concept["id"],
            properties=[
                UpdateRecord.PropertyDiff(
                    name="name",
                    data_type=dt.String(),
                    old_value="Alice",
                    new_value="Bob",
                ),
                UpdateRecord.PropertyDiff(
                    name="age", data_type=dt.Long(), old_value=25, new_value=39
                ),
                UpdateRecord.PropertyDiff(
                    name="occupation",
                    data_type=dt.String(),
                    old_value="Student",
                    new_value="Teacher",
                ),
            ],
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Update values again, removing a non-required property
    r = put(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{record_id}",
        concept_id=concept["id"],
        record_id=record["id"],
        json={
            "values": [
                {"name": "occupation", "value": "Teacher"},
                {"name": "name", "value": "Bob"},
            ]
        },
    )
    assert r.status_code == 200
    record = r.json
    values = record["values"]
    assert len(values) == 3
    assert values[0]["name"] == "name"
    assert values[0]["value"] == "Bob"
    assert values[1]["name"] == "age"
    assert values[1]["value"] is None
    assert values[2]["name"] == "occupation"
    assert values[2]["value"] == "Teacher"

    # Delete the record
    r = delete(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{record_id}",
        concept_id=concept["id"],
        record_id=record["id"],
    )
    assert r.status_code == 200

    # Emit DeleteRecord
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=DeleteRecord(id=UUID(record["id"]), name="Bob", model_id=concept["id"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # No longer visible
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{record_id}",
        concept_id=concept["id"],
        record_id=record["id"],
    )
    assert r.status_code == 404

    # Make sure auditing messages were emitted:
    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 11


def test_concept_instance_data_type_serialization(
    concept, create_property, create_concept_instance, put
):

    # Create a property
    r = put(
        "/v1/datasets/{dataset}/concepts/{concept_id}/properties",
        concept_id=concept["id"],
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "locked": False,
                "conceptTitle": True,
                "required": False,
                "default": False,
            },
            {
                "name": "dob",
                "displayName": "Dob",
                "description": "",
                "dataType": "Date",
                "locked": False,
                "conceptTitle": False,
                "required": False,
                "default": True,
            },
            {
                "name": "favorite_dates",
                "displayName": "Favorite_Dates",
                "description": "",
                "dataType": {"type": "array", "items": {"type": "Date"}},
                "locked": False,
                "conceptTitle": False,
                "required": False,
                "default": True,
            },
        ],
    )
    assert r.status_code == 200

    created = create_concept_instance(
        concept,
        [
            {"name": "name", "value": "Bob"},
            {"name": "dob", "value": "2004-05-05T00:00:00"},
            {
                "name": "favorite_dates",
                "value": ["2004-05-05 00:00:00", "1999-01-03T00:00:00+10:00"],
            },
        ],
    )

    assert [(p["name"], p["value"]) for p in created["values"]] == [
        ("name", "Bob"),
        ("dob", "2004-05-05T00:00:00"),
        ("favorite_dates", ["2004-05-05T00:00:00", "1999-01-02T14:00:00"]),
    ]


def test_concept_instance_pagination(get, concept, create_concept_instance):
    ci1 = create_concept_instance(concept, [])  # noqa: F841
    ci2 = create_concept_instance(concept, [])
    ci3 = create_concept_instance(concept, [])

    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
        query_string={"limit": 1, "offset": 1},
    )
    assert r.json == [ci2]

    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
        query_string={"limit": 1, "offset": 2},
    )
    assert r.json == [ci3]


def test_concept_instance_dont_sort_many_records(
    get, concept, create_concept_instance, partitioned_db
):
    NUM_RECORDS = 5
    MAX_RECORDS = 2
    records = [create_concept_instance(concept, []) for i in range(NUM_RECORDS)]

    with partitioned_db.transaction() as tx:
        model = partitioned_db.get_model(concept["id"])
        # Get records ordered by Neo4j id
        nodes = tx.run(
            f"""
            MATCH ({labels.record("r")})
            WHERE r.`@id` IN {[str(record["id"]) for record in records]}
            RETURN r.`@id` AS id
            """
        ).records()
        record_ids = [node["id"] for node in nodes]
        r = partitioned_db.get_all_records_offset_tx(
            tx, model=model, order_by=None, max_records=MAX_RECORDS
        )
        # This endpoint is not ordered when there are too many records, so a manual
        # sort is necessary here to check the correctness of the values
        assert sorted([str(record.id) for record in r]) == sorted(record_ids)

        # Cannot pass in order column or direction if record count is greater than maxRecords
        with pytest.raises(CannotSortRecordsError) as e:
            r = partitioned_db.get_all_records_offset_tx(
                tx,
                model=model,
                order_by=OrderByField(name="created_at", ascending=True),
                max_records=MAX_RECORDS,
            )
            # Access generator to raise error
            sorted_records = list(r)
        assert CannotSortRecordsError(NUM_RECORDS, MAX_RECORDS).message in str(e.value)


def test_concept_instance_bulk_deletion(
    get,
    delete,
    concept,
    create_concept_instance,
    create_property,
    trace_id,
    jobs_client,
    valid_organization,
    valid_dataset,
    valid_user,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user
    create_property(concept["id"], "name", title=True)

    ci1 = create_concept_instance(concept, [{"name": "name", "value": "Alice"}])
    ci2 = create_concept_instance(concept, [{"name": "name", "value": "Bob"}])
    ci3 = create_concept_instance(concept, [{"name": "name", "value": "Charlie"}])

    r = delete(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
        json=[ci2["id"], ci1["id"], "an-invalid-instance-id"],
    )
    assert r.status_code == 200
    assert r.json["success"] == [ci2["id"], ci1["id"]]
    assert r.json["errors"] == [
        ["an-invalid-instance-id", "Could not delete an-invalid-instance-id"]
    ]

    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances",
        concept_id=concept["id"],
    )
    assert r.status_code == 200
    assert r.json == [ci3]

    jobs_client.send_changelog_events.assert_called_with(
        dataset_id=dataset_id.id,
        events=[
            DeleteRecord(id=UUID(ci2["id"]), name="Bob", model_id=concept["id"]),
            DeleteRecord(id=UUID(ci1["id"]), name="Alice", model_id=concept["id"]),
        ],
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )


def test_create_concept_relationship(post, create_concept, valid_dataset):
    c1 = create_concept(name="patient", display_name="Patient", description="A patient")
    c2 = create_concept(name="doctor", display_name="Doctor", description="A doctor")

    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
            "from": c1["id"],
            "to": c2["id"],
        },
    )
    assert r.status_code == 201
    assert drop_volatile(r.json) == {
        "name": "a_relationship",
        "displayName": "A Relationship",
        "description": "A description",
        "schema": [],
        "from": c1["id"],
        "to": c2["id"],
    }

    # Cannot create another relationship with the same name between same models
    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "",
            "description": "",
            "schema": [],
            "from": c1["id"],
            "to": c2["id"],
        },
    )
    assert r.status_code == 400

    # But can create one with the same name in the reverse direction
    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "",
            "description": "",
            "schema": [],
            "from": c2["id"],
            "to": c1["id"],
        },
    )
    assert r.status_code == 201

    # This new one cannot be duplicated
    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "",
            "description": "",
            "schema": [],
            "from": c2["id"],
            "to": c1["id"],
        },
    )
    assert r.status_code == 400

    # Can handle dashes in relationship names
    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "is-about",
            "displayName": "Is About",
            "description": "",
            "schema": [],
            "from": c1["id"],
            "to": c2["id"],
        },
    )
    assert r.status_code == 201
    assert drop_volatile(r.json) == {
        "name": "is-about",
        "displayName": "Is About",
        "description": "",
        "schema": [],
        "from": c1["id"],
        "to": c2["id"],
    }


def test_create_concept_relationship_stub(post, get, create_concept):
    c1 = create_concept(name="patient", display_name="Patient", description="A patient")
    c2 = create_concept(name="doctor", display_name="Doctor", description="A doctor")

    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
        },
    )
    assert r.status_code == 201
    assert drop_volatile(r.json) == {
        "name": "a_relationship",
        "displayName": "A Relationship",
        "description": "A description",
        "schema": [],
        "from": None,
        "to": None,
    }

    r = get(
        "/v1/datasets/{dataset}/relationships/{id_}",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
        },
        id_=r.json["id"],
    )
    assert r.status_code == 200
    assert drop_volatile(r.json) == {
        "name": "a_relationship",
        "displayName": "A Relationship",
        "description": "A description",
        "schema": [],
        "from": None,
        "to": None,
    }


def test_delete_concept_relationship_stub(post, delete, create_concept):
    c1 = create_concept(name="patient", display_name="Patient", description="A patient")
    c2 = create_concept(name="doctor", display_name="Doctor", description="A doctor")

    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
        },
    )
    assert r.status_code == 201
    assert drop_volatile(r.json) == {
        "name": "a_relationship",
        "displayName": "A Relationship",
        "description": "A description",
        "schema": [],
        "from": None,
        "to": None,
    }

    r = delete("/v1/datasets/{dataset}/relationships/{id_}", id_=r.json["id"])
    assert r.status_code == 200


def test_create_concept_relationship_stub_in_empty_dataset(post):
    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
        },
    )
    assert r.status_code == 201
    assert drop_volatile(r.json) == {
        "name": "a_relationship",
        "displayName": "A Relationship",
        "description": "A description",
        "schema": [],
        "from": None,
        "to": None,
    }


def test_cannot_create_duplicate_concept_relationship_stub(post):
    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
        },
    )
    assert r.status_code == 201

    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
        },
    )
    assert r.status_code == 400


def test_get_concept_relationships_from_stub(
    get,
    create_concept,
    create_concept_relationship,
    create_concept_instance,
    create_concept_instance_relationship,
):
    # concepts

    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )

    # relationship stub
    sees = create_concept_relationship("sees", "Sees", None, None)  # noqa: F841

    p = create_concept_instance(patient, [])
    d = create_concept_instance(doctor, [])

    ir = create_concept_instance_relationship(p["id"], sees["name"], d["id"])

    r = get("/v1/datasets/{dataset}/relationships/sees/instances")

    assert r.status_code == 200
    r = r.json
    assert len(r) == 1
    assert r[0]["from"] == p["id"]
    assert r[0]["to"] == d["id"]
    assert r[0]["schemaRelationshipId"] == ir[0]["schemaRelationshipId"]


def test_get_concept_relationships(
    random_uuid,
    get,
    create_concept,
    create_concept_relationship,
    create_linked_property,
):
    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship(  # noqa: F841
        "sees", "Sees", patient["id"], doctor["id"]
    )
    treats = create_concept_relationship(  # noqa: F841
        "treats", "Treats", doctor["id"], patient["id"]
    )
    attends = create_concept_relationship(  # noqa: F841
        "attends", "Attends", doctor["id"], visit["id"]
    )

    # relationship stub
    belongs_to = create_concept_relationship(  # noqa: F841
        "belongs_to", "Belongs To", None, None
    )

    # linked property
    mentor = create_linked_property(doctor, "mentor", doctor)

    # Include stubs in full response; exclude linked properties
    r = get("/v1/datasets/{dataset}/relationships")
    assert r.status_code == 200
    assert len(r.json) == 4

    # Get with "from" present:
    # random - none found
    r = get("/v1/datasets/{dataset}/relationships", query_string={"from": random_uuid})
    assert r.status_code == 200
    r = r.json
    assert len(r) == 0

    r = get("/v1/datasets/{dataset}/relationships", query_string={"from": doctor["id"]})
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2
    for k in r:
        assert k["from"] == doctor["id"]

    # Get with "to" present:
    # random - none found
    r = get("/v1/datasets/{dataset}/relationships", query_string={"to": random_uuid})
    assert r.status_code == 200
    r = r.json
    assert len(r) == 0

    r = get("/v1/datasets/{dataset}/relationships", query_string={"to": patient["id"]})
    assert r.status_code == 200
    r = r.json
    assert len(r) == 1
    assert r[0]["to"] == patient["id"]

    # Both
    r = get(
        "/v1/datasets/{dataset}/relationships",
        query_string={"from": doctor["id"], "to": visit["id"]},
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 1
    assert r[0]["from"] == doctor["id"]
    assert r[0]["to"] == visit["id"]


def test_get_concept_relationship(
    random_uuid, get, create_concept, create_concept_relationship, valid_user
):
    _, user_node_id = valid_user

    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="VBisit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    treats = create_concept_relationship(  # noqa: F841
        "treats", "Treats", doctor["id"], patient["id"]
    )
    attends = create_concept_relationship(  # noqa: F841
        "attends", "attends", doctor["id"], visit["id"]
    )

    # Not found = 404
    r = get("/v1/datasets/{dataset}/relationships/{id_}", id_=random_uuid)
    assert r.status_code == 404

    r = get("/v1/datasets/{dataset}/relationships/{id_}", id_=sees["id"])
    assert r.status_code == 200
    r = r.json
    assert r["id"] == sees["id"]
    assert r["name"] == "sees"
    assert r["createdBy"] == user_node_id
    assert r["updatedBy"] == user_node_id

    # Can get via relationship name
    r = get("/v1/datasets/{dataset}/relationships/sees")
    assert r.status_code == 200
    assert r.json["id"] == sees["id"]
    assert r.json["name"] == "sees"


def test_update_concept_relationship(
    random_uuid, get, put, create_concept, create_concept_relationship
):
    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    attends = create_concept_relationship(
        "attends", "Attends", doctor["id"], visit["id"]
    )

    r = get("/v1/datasets/{dataset}/relationships/{id_}", id_=sees["id"])
    assert r.status_code == 200
    assert r.json["displayName"] == sees["displayName"]

    # Try to update a nonexistent relationship:
    r = put(
        "/v1/datasets/{dataset}/relationships/{id_}",
        id_=random_uuid,  # random
        json={"displayName": "Sees (UPDATED)"},
    )
    assert r.status_code == 404

    r = put(
        "/v1/datasets/{dataset}/relationships/{id_}",
        id_=sees["id"],
        json={"displayName": "Sees (UPDATED)"},
    )
    assert r.status_code == 200
    assert r.json[0]["displayName"] == "Sees (UPDATED)"

    r = get("/v1/datasets/{dataset}/relationships/{id_}", id_=sees["id"])
    assert r.status_code == 200
    assert r.json["displayName"] == "Sees (UPDATED)"

    r = get("/v1/datasets/{dataset}/relationships/{id_}", id_=attends["id"])
    assert r.status_code == 200
    assert r.json["displayName"] == attends["displayName"]

    # Update via relationship name
    r = put(
        "/v1/datasets/{dataset}/relationships/sees",
        id_=sees["id"],
        json={"displayName": "Sees (UPDATED AGAIN)"},
    )
    assert r.status_code == 200
    assert r.json[0]["displayName"] == "Sees (UPDATED AGAIN)"


def test_delete_concept_relationship(
    get, delete, random_uuid, create_concept, create_concept_relationship
):
    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    attends = create_concept_relationship(  # noqa: F841
        "attends", "Attends", doctor["id"], visit["id"]
    )

    # Try deleting a non-existent relationship:
    r = delete("/v1/datasets/{dataset}/relationships/{id_}", id_=random_uuid)
    assert r.status_code == 404

    # Exists
    r = get("/v1/datasets/{dataset}/relationships/{id_}", id_=sees["id"])
    assert r.status_code == 200
    assert r.json["id"] == sees["id"]

    # Delete
    r = delete("/v1/datasets/{dataset}/relationships/{id_}", id_=sees["id"])
    assert r.status_code == 200
    assert r.json == [sees["id"]]

    # Doesn't exist
    r = get("/v1/datasets/{dataset}/relationships/{id_}", id_=sees["id"])
    assert r.status_code == 404


def test_create_concept_instance_relationship_batch_from_stub(
    get,
    create_concept,
    create_concept_instance,
    create_concept_relationship_stub,
    create_concept_instance_relationship_batch,
    create_property,
):
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")
    create_property(visit["id"], "field")
    patient_attends_visit = create_concept_relationship_stub(  # noqa: F841
        "attends", "Attends"
    )
    p = create_concept_instance(patient, [])
    v = [
        create_concept_instance(visit, [{"name": "field", "value": str(i)}])
        for i in range(3)
    ]  # noqa: F841
    p_array = [p for i in range(3)]

    r = create_concept_instance_relationship_batch(p_array, "attends", v)

    assert r.status_code == 200
    assert len(r.json) == 3
    p_attends_v0 = r.json[0]
    p_attends_v1 = r.json[1]
    p_attends_v2 = r.json[2]
    assert p_attends_v0[1]["name"] == patient_attends_visit["name"]
    assert p_attends_v0[0]["from"] == p["id"]
    assert p_attends_v0[0]["to"] == v[0]["id"]
    assert p_attends_v1[1]["name"] == patient_attends_visit["name"]
    assert p_attends_v1[0]["from"] == p["id"]
    assert p_attends_v1[0]["to"] == v[1]["id"]
    assert p_attends_v2[1]["name"] == patient_attends_visit["name"]
    assert p_attends_v2[0]["from"] == p["id"]
    assert p_attends_v2[0]["to"] == v[2]["id"]


def test_create_concept_instance_relationship(
    post,
    create_concept,
    create_concept_instance,
    create_concept_relationship,
    create_concept_instance_relationship_batch,
):
    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    doctor_attends_visit = create_concept_relationship(  # noqa: F841
        "attends", "Attends", doctor["id"], visit["id"]
    )

    # Batch large numbers of relationships together
    p = [create_concept_instance(patient, []) for i in range(200)]
    d = [create_concept_instance(doctor, []) for i in range(200)]
    v = [create_concept_instance(visit, []) for i in range(200)]  # noqa: F841

    rs = create_concept_instance_relationship_batch(p, sees["id"], d)

    assert len(rs.json) == 200
    p_sees_d = rs.json[0][0]
    assert p_sees_d["from"] == p[0]["id"]
    assert p_sees_d["to"] == d[0]["id"]
    assert rs.status_code == 200

    # Creation fails for records that don't have a model relationship:
    r = post(
        "/v1/datasets/{dataset}/relationships/{id_}/instances/batch",
        id_=sees["id"],
        json=[{"from": p[0]["id"], "to": v[0]["id"], "values": []}],
    )
    assert r.status_code == 400

    # Creation succeeds:
    r = post(
        "/v1/datasets/{dataset}/relationships/{id_}/instances/batch",
        id_=sees["id"],
        json=[{"from": p[0]["id"], "to": d[0]["id"], "values": []}],
    )

    assert r.status_code == 200
    p_sees_d = r.json[0][0]
    assert p_sees_d["from"] == p[0]["id"]
    assert p_sees_d["to"] == d[0]["id"]

    # Creation is idempotent
    # Creating duplicate relationship returns the original and updates timestamps
    r = post(
        "/v1/datasets/{dataset}/relationships/{id_}/instances/batch",
        id_=sees["id"],
        json=[{"from": p[0]["id"], "to": d[0]["id"], "values": []}],
    )

    assert r.status_code == 200
    new_p_sees_d = r.json[0][0]
    assert new_p_sees_d.pop("updatedAt") > p_sees_d.pop("updatedAt")
    assert new_p_sees_d == p_sees_d

    # Batch allows different model relationships with same relationship name in
    # same operation

    patient_attends_visit = create_concept_relationship(  # noqa: F841
        "attends", "Attends", patient["id"], visit["id"]
    )

    p_1 = create_concept_instance(patient, [])
    d_1 = create_concept_instance(doctor, [])
    v_1 = create_concept_instance(visit, [])

    r = post(
        "/v1/datasets/{dataset}/relationships/attends/instances/batch",
        id_=patient_attends_visit["id"],
        json=[
            {"from": p_1["id"], "to": v_1["id"], "values": []},
            {"from": d_1["id"], "to": v_1["id"], "values": []},
        ],
    )

    assert r.status_code == 200
    assert len(r.json) == 2
    p1_attends_v1 = r.json[0]
    d1_attends_v1 = r.json[1]

    assert p1_attends_v1[1]["id"] == patient_attends_visit["id"]
    assert p1_attends_v1[0]["from"] == p_1["id"]
    assert p1_attends_v1[0]["to"] == v_1["id"]

    assert d1_attends_v1[1]["id"] == doctor_attends_visit["id"]
    assert d1_attends_v1[0]["from"] == d_1["id"]
    assert d1_attends_v1[0]["to"] == v_1["id"]


def test_get_concept_instance_relationship(
    get,
    create_concept,
    create_concept_instance,
    create_concept_instance_relationship,
    create_concept_relationship,
    valid_user,
):
    _, user_node_id = valid_user

    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    attends = create_concept_relationship(  # noqa: F841
        "attends", "Attends", doctor["id"], visit["id"]
    )
    # create instances
    p = [create_concept_instance(patient, []), create_concept_instance(patient, [])]
    d = [create_concept_instance(doctor, []), create_concept_instance(doctor, [])]
    v = [  # noqa: F841
        create_concept_instance(visit, []),
        create_concept_instance(visit, []),
    ]

    # create instances relationships
    ri = [
        create_concept_instance_relationship(pi["id"], sees["name"], di["id"])
        for pi, di in zip(p, d)
    ]

    # Fetch - attend
    r = get(
        "/v1/datasets/{dataset}/relationships/sees/instances/{id_}", id_=ri[0][0]["id"]
    )

    assert r.status_code == 200
    r = r.json
    assert r["id"] == ri[0][0]["id"]
    assert r["createdBy"] == user_node_id
    assert r["updatedBy"] == user_node_id


def test_get_concept_instance_relationships(
    get,
    create_concept,
    create_concept_instance,
    create_concept_relationship,
    create_concept_instance_relationship,
):
    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    attends = create_concept_relationship(
        "attends", "Attends", doctor["id"], visit["id"]
    )

    # create instances
    p = [create_concept_instance(patient, []), create_concept_instance(patient, [])]
    d = [create_concept_instance(doctor, []), create_concept_instance(doctor, [])]
    v = [  # noqa: F841
        create_concept_instance(visit, []),
        create_concept_instance(visit, []),
    ]

    # create instances relationships
    for pi, di in zip(p, d):
        create_concept_instance_relationship(pi["id"], sees["name"], di["id"])

    # Fetch - attend
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=attends["id"],
    )
    assert r.status_code == 200
    assert len(r.json) == 0

    # Fetch - sees
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=sees["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2


def test_delete_single_concept_instance_relationship(
    random_uuid,
    get,
    delete,
    create_concept,
    create_concept_instance,
    create_concept_relationship,
    create_concept_instance_relationship,
):
    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    attends = create_concept_relationship(  # noqa: F841
        "attends", "Attends", doctor["id"], visit["id"]
    )
    # create instances
    p = [create_concept_instance(patient, []), create_concept_instance(patient, [])]
    d = [create_concept_instance(doctor, []), create_concept_instance(doctor, [])]
    v = [  # noqa: F841
        create_concept_instance(visit, []),
        create_concept_instance(visit, []),
    ]

    # create instances relationships
    sees_ri = [
        create_concept_instance_relationship(pi["id"], sees["name"], di["id"])
        for pi, di in zip(p, d)
    ]
    sees_ri_ids = [rel[0]["id"] for rel in sees_ri]
    attends_ri = [  # noqa: F841
        create_concept_instance_relationship(di["id"], attends["name"], vi["id"])
        for di, vi in zip(d, v)
    ]

    # Fetch - sees
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=sees["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2

    # Fetch - attends
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=attends["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2

    # Deletion a nonexistent relationship should 404:
    r = delete(
        "/v1/datasets/{dataset}/relationships/{rid}/instances/{id_}",
        rid=sees["id"],
        id_=random_uuid,
    )
    assert r.status_code == 404

    # Delete a single "sees" record relationship:
    r = delete(
        "/v1/datasets/{dataset}/relationships/{rid}/instances/{id_}",
        rid=sees["id"],
        id_=sees_ri_ids[0],
    )
    assert r.status_code == 200
    assert r.json == sees_ri_ids[0]  # We get back the ID of the deleted record

    # "sees" should be 1 less:
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=sees["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 1
    assert sees_ri_ids[0] not in r

    # "attends" shouldn't be empty
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=attends["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2


def test_delete_concept_instance_relationships(
    get,
    delete,
    create_concept,
    create_concept_instance,
    create_concept_relationship,
    create_concept_instance_relationship,
):
    # concepts
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )
    visit = create_concept(name="visit", display_name="Visit", description="A visit")

    # relationships
    sees = create_concept_relationship("sees", "Sees", patient["id"], doctor["id"])
    attends = create_concept_relationship(  # noqa: F841
        "attends", "Attends", doctor["id"], visit["id"]
    )

    # create instances
    p = [create_concept_instance(patient, []), create_concept_instance(patient, [])]
    d = [create_concept_instance(doctor, []), create_concept_instance(doctor, [])]
    v = [  # noqa: F841
        create_concept_instance(visit, []),
        create_concept_instance(visit, []),
    ]

    # create instances relationships
    sees_ri = [
        create_concept_instance_relationship(pi["id"], sees["name"], di["id"])
        for pi, di in zip(p, d)
    ]
    attends_ri = [  # noqa: F841
        create_concept_instance_relationship(di["id"], attends["name"], vi["id"])
        for di, vi in zip(d, v)
    ]

    # Fetch - sees
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=sees["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2

    # Fetch - attends
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=attends["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2

    sees_ri_ids = [rel[0]["id"] for rel in sees_ri]

    # Delete "sees" record relationships:
    r = delete(
        "/v1/datasets/{dataset}/relationships/instances/bulk",
        json={"relationshipInstanceIds": sees_ri_ids},
    )
    assert r.status_code == 200
    r = r.json
    for id_ in sees_ri_ids:
        assert id_ in r

    # "sees" relationships -- should be empty
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=sees["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 0

    # "attends" relationships -- shouldn't be empty
    r = get(
        "/v1/datasets/{dataset}/relationships/{rid}/instances",  # name or ID works
        rid=attends["id"],
    )
    assert r.status_code == 200
    r = r.json
    assert len(r) == 2


def test_linked_properties(
    movie_db,
    configure_get,
    configure_post,
    configure_put,
    configure_delete,
    audit_logger,
    create_concept,
    create_property,
    create_concept_instance,
    get_concept,
    valid_user,
):
    db, _ = movie_db
    _, user_node_id = valid_user

    get = configure_get(audit_logger)
    post = configure_post(audit_logger)
    put = configure_put(audit_logger)
    delete = configure_delete(audit_logger)

    # Get all schema (model) linked properties:
    r = get("/v1/datasets/{dataset}/concepts/linked/properties")
    assert r.status_code == 200
    linked_properties = r.json
    assert len(linked_properties) == 2

    # Get schema linked properties for a given concept (model):
    for src_id in set(p["concept"] for p in linked_properties):
        r = get("/v1/datasets/{dataset}/concepts/{id_}/linked", id_=src_id)
        assert r.status_code == 200
        p = r.json
        for pp in p:
            assert src_id == pp["concept"] and src_id == pp["link"]["from"]

    # Concepts
    movie = get_concept("movie")
    genre = get_concept("genre")
    person = get_concept("person")

    # get all genres:
    genres = db.get_all_records("genre")

    # get all movies:
    movies = db.get_all_records("movie")

    # pick out "Top Gun" and "The Matrix"
    top_gun = [m for m in movies if m.values["title"] == "Top Gun"].pop()
    the_matrix = [m for m in movies if m.values["title"] == "The Matrix"].pop()

    action = [g for g in genres if g.values["name"] == "Action"].pop()

    studio = create_concept("studio", "Studio", "Studio")
    create_property(studio["id"], "name")
    _20th_century_fox = create_concept_instance(  # noqa: F841
        studio, [{"name": "name", "value": "20th Century Fox"}]
    )
    warner_bros = create_concept_instance(  # noqa: F841
        studio, [{"name": "name", "value": "Warner Bros"}]
    )
    paramount_pictures = create_concept_instance(  # noqa: F841
        studio, [{"name": "name", "value": "Paramount Pictures"}]
    )
    columbia_pictures = create_concept_instance(  # noqa: F841
        studio, [{"name": "name", "value": "Columbia Pictures"}]
    )
    universal_pictures = create_concept_instance(  # noqa: F841
        studio, [{"name": "name", "value": "Universal Pictures"}]
    )
    walt_disney_pictures = create_concept_instance(  # noqa: F841
        studio, [{"name": "name", "value": "Walt Disney Pictures"}]
    )

    # We should start with 2 model linked properties on "person":
    r = get("/v1/datasets/{dataset}/concepts/{id_}/linked", id_=person["id"])
    assert len(r.json) == 2

    # Create a new schema (model) linked property
    r = post(
        "/v1/datasets/{dataset}/concepts/{id_}/linked",
        id_=movie["id"],
        json={"name": "financed_by", "displayName": "Financed by", "to": studio["id"]},
    )
    assert r.status_code == 201
    financed_by = r.json
    assert (
        financed_by["name"] == "financed_by"
        and financed_by["from"] == movie["id"]
        and financed_by["to"] == studio["id"]
        and financed_by["createdBy"] == user_node_id
        and financed_by["updatedBy"] == user_node_id
    )

    # Get a new count:
    r = get("/v1/datasets/{dataset}/concepts/{id_}/linked", id_=movie["id"])
    assert r.status_code == 200
    assert len(r.json) == 1

    # Update a new schema (model) linked property (also the concept ID is ignored in the URL)
    r = put(
        "/v1/datasets/{dataset}/concepts/{id_}/linked/{link_id}",
        id_=None,
        link_id=financed_by["id"],
        json={"displayName": f"!!!{financed_by['name']}!!!", "position": 77},
    )
    assert r.status_code == 200
    r = r.json
    assert r["displayName"] == f"!!!{financed_by['name']}!!!"
    assert r["position"] == 77

    # Creating a linked property by an invalid name/type should fail (ignore the concept ID in the URL):
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,  # ignore
        id_=top_gun.id,
        json={"name": "foo", "to": paramount_pictures["id"]},
    )
    assert r.status_code == 400

    # Creating a linked property to an invalid target should fail (ignore the concept ID in the URL):
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,  # ignore
        id_=top_gun.id,
        json={
            "name": "financed_by",
            "schemaLinkedPropertyId": financed_by["id"],
            "to": movie["id"],
        },
    )
    assert r.status_code == 400

    # Creating a linked property by an valid name/type should succeed (ignore the concept ID in the URL):
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=movie["id"],  # ignore
        id_=top_gun.id,
        json={
            "name": "financed_by",
            "schemaLinkedPropertyId": financed_by["id"],
            "to": paramount_pictures["id"],
        },
    )
    assert r.status_code == 201
    lp = r.json
    assert (
        lp["from"] == str(top_gun.id)
        and lp["to"] == paramount_pictures["id"]
        and lp["createdBy"] == user_node_id
        and lp["updatedBy"] == user_node_id
        and lp["schemaLinkedPropertyId"] == financed_by["id"]
    )

    # Verify that a link exists for "top gun" (ignore concept_id):
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,
        id_=str(top_gun.id),
    )
    assert r.status_code == 200
    good_lps = r.json

    # linked records should exist for "financed_by"
    assert len(good_lps) == 1
    assert (
        good_lps[0]["displayName"] == "!!!financed_by!!!"
        and good_lps[0]["from"] == str(top_gun.id)
        and good_lps[0]["to"] == paramount_pictures["id"]
    )

    # Delete a schema linked property (also the concept ID is ignored in the URL)
    # (should fail since record linked property exists)
    r = delete(
        "/v1/datasets/{dataset}/concepts/{id_}/linked/{link_id}",
        id_=None,  # ignore
        link_id=financed_by[
            "id"
        ],  # (m)-[:RELATED_TO { type: "financed_by" }]->(n) does not exist
    )
    assert r.status_code == 400
    assert "violation" in r.json["message"]

    # Deleting a nonexistent a record linked property should fail (not found):
    r = delete(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/{link_id}",
        concept_id=person["id"],  # ignore
        id_=person["id"],  # ignore
        link_id=movie["id"],  # invalid
    )
    assert r.status_code == 404

    # Deleting a valid record linked property should succeed (ignore the concept)
    r = delete(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/{link_id}",
        concept_id=None,  # ignore
        id_=top_gun.id,
        link_id=good_lps[0]["id"],
    )
    assert r.status_code == 200

    # Verify we still have 1 schema linked properties linked to the model "movie":
    r = get("/v1/datasets/{dataset}/concepts/{id_}/linked", id_=movie["id"])
    assert r.status_code == 200
    # Should be "financed_by"
    assert len(r.json) == 1
    assert set([p["link"]["name"] for p in r.json]) == set(["financed_by"])

    # Deleting the schema linked property should succeed now:
    # Delete a schema linked property (also the concept ID is ignored in the URL)
    # (should fail since record linked property exists)
    r = delete(
        "/v1/datasets/{dataset}/concepts/{id_}/linked/{link_id}",
        id_=person["id"],  # ignore
        link_id=financed_by["id"],
    )
    assert r.status_code == 200

    # There should only be no schema linked properties on "movie"
    r = get("/v1/datasets/{dataset}/concepts/{id_}/linked", id_=movie["id"])
    assert r.status_code == 200
    assert len(r.json) == 0

    # Create 2 new schema linked properties (recreaye "financed_by" + create "most_famous_actor") on "movie"
    r = post(
        "/v1/datasets/{dataset}/concepts/{id_}/linked/bulk",
        id_=movie["id"],
        json=[
            {
                "name": "financed_by",
                "displayName": "Financed by",
                "to": studio["id"],
                "position": 5,
            },
            {
                "name": "most_famous_actor",
                "displayName": "Most famous actor",
                "to": person["id"],
                "position": 6,
            },
        ],
    )
    assert r.status_code == 201

    # BATCH creation tests #

    # Get the new ids:
    r = get("/v1/datasets/{dataset}/concepts/{id_}/linked", id_=movie["id"])

    assert r.status_code == 200
    assert len(r.json) == 2

    financed_by = [x["link"] for x in r.json if x["link"]["name"] == "financed_by"][0]
    most_famous_actor = [
        x["link"] for x in r.json if x["link"]["name"] == "most_famous_actor"
    ][0]

    # For records, verify that no links exists for "Top Gun" (ignore concept_id):
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,
        id_=str(top_gun.id),
    )
    assert r.status_code == 200
    lps = r.json
    assert len(lps) == 0

    # If a model relationship doesn't exist, the whole request fails:
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/batch",
        concept_id=movie["id"],  # ignore
        id_=top_gun.id,
        json={
            "data": [
                {
                    "name": "financed_by",
                    "schemaLinkedPropertyId": financed_by["id"],
                    "to": paramount_pictures["id"],
                },
                {
                    "name": "financed_by",
                    "schemaLinkedPropertyId": "badSchemaLinkedPropertyId",
                    "to": paramount_pictures["id"],
                },
            ]
        },
    )

    assert r.status_code == 400
    assert "no model linked property exists" in r.json["detail"]

    # Verify that only link exists for "Top Gun" ("financed_by") (ignore concept_id):
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,
        id_=str(top_gun.id),
    )
    assert r.status_code == 200
    lps = r.json
    assert len(lps) == 0

    # if a record doesn't exist, the whole request fails
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/batch",
        concept_id=movie["id"],  # ignore
        id_=top_gun.id,
        json={
            "data": [
                {
                    "name": "financed_by",
                    "schemaLinkedPropertyId": financed_by["id"],
                    "to": paramount_pictures["id"],
                },
                {
                    "name": "most_famous_actor",
                    "schemaLinkedPropertyId": most_famous_actor["id"],
                    "to": paramount_pictures["id"],
                },
            ]
        },
    )

    assert r.status_code == 404
    assert "violation" in r.json["message"]

    # Verify that no link exists for "top gun" (ignore concept_id):
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,
        id_=str(top_gun.id),
    )
    assert r.status_code == 200
    lps = r.json
    assert len(lps) == 0

    tom_hanks = create_concept_instance(  # noqa: F841
        person, [{"name": "name", "value": "Tom Hanks"}]
    )

    # if duplicate schemaLinkedProperties are used the whole request fails
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/batch",
        concept_id=movie["id"],  # ignore
        id_=top_gun.id,
        json={
            "data": [
                {
                    "name": "financed_by",
                    "schemaLinkedPropertyId": financed_by["id"],
                    "to": paramount_pictures["id"],
                },
                {
                    "name": "financed_by",
                    "schemaLinkedPropertyId": financed_by["id"],
                    "to": columbia_pictures["id"],
                },
            ]
        },
    )

    assert r.status_code == 400
    assert "duplicate model linked properties" in r.json["detail"]

    # Verify that no link exists for "top gun" (ignore concept_id):
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,
        id_=str(top_gun.id),
    )
    assert r.status_code == 200
    lps = r.json
    assert len(lps) == 0

    # can bulk create if everything is valid
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/batch",
        concept_id=movie["id"],  # ignore
        id_=top_gun.id,
        json={
            "data": [
                {
                    "name": "financed_by",
                    "schemaLinkedPropertyId": financed_by["id"],
                    "to": paramount_pictures["id"],
                },
                {
                    "name": "most_famous_actor",
                    "schemaLinkedPropertyId": most_famous_actor["id"],
                    "to": tom_hanks["id"],
                },
            ]
        },
    )

    assert r.status_code == 200
    assert len(r.json["data"]) == 2

    # Verify that two links exist for "top gun" (ignore concept_id):
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,
        id_=str(top_gun.id),
    )
    assert r.status_code == 200
    lps = r.json
    assert len(lps) == 2

    wilson = create_concept_instance(  # noqa: F841
        person, [{"name": "name", "value": "Wilson"}]
    )

    # can update the "to" record on existing linked properties
    r = post(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/batch",
        concept_id=movie["id"],  # ignore
        id_=top_gun.id,
        json={
            "data": [
                {
                    "name": "financed_by",
                    "schemaLinkedPropertyId": financed_by["id"],
                    "to": universal_pictures["id"],
                },
                {
                    "name": "most_famous_actor",
                    "schemaLinkedPropertyId": most_famous_actor["id"],
                    "to": wilson["id"],
                },
            ]
        },
    )

    assert r.status_code == 200

    # Verify that two UPDATED links exist for "top gun" (ignore concept_id):
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked",
        concept_id=None,
        id_=str(top_gun.id),
    )
    assert r.status_code == 200
    lps = r.json
    assert len(lps) == 2

    assert lps[0]["to"] == universal_pictures["id"]
    assert lps[1]["to"] == wilson["id"]

    # cleanup
    r = delete(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/{link_id}",
        concept_id=person["id"],  # ignore
        id_=top_gun.id,
        link_id=lps[0]["id"],
    )
    assert r.status_code == 200

    r = delete(
        "/v1/datasets/{dataset}/concepts/{concept_id}/instances/{id_}/linked/{link_id}",
        concept_id=person["id"],  # ignore
        id_=top_gun.id,
        link_id=lps[1]["id"],
    )
    assert r.status_code == 200

    r = get("/v1/datasets/{dataset}/concepts/{id_}/linked", id_=movie["id"])
    assert [drop_volatile(x["link"]) for x in r.json] == [
        {
            "name": "financed_by",
            "displayName": "Financed by",
            "from": movie["id"],
            "to": studio["id"],
            "type": "schemaLinkedProperty",
            "position": 5,
        },
        {
            "name": "most_famous_actor",
            "displayName": "Most famous actor",
            "from": movie["id"],
            "to": person["id"],
            "type": "schemaLinkedProperty",
            "position": 6,
        },
    ]


def test_self_referential_linked_properties(
    sample_patient_db, create_linked_property, create_linked_property_instance
):
    patient = sample_patient_db["models"]["patient"]
    bob = sample_patient_db["records"]["bob"]

    looks_like = create_linked_property(patient, "looks_like", patient)

    bob_looks_like_bob = create_linked_property_instance(patient, bob, looks_like, bob)

    assert drop_volatile(bob_looks_like_bob) == {
        "from": str(bob.id),
        "to": str(bob.id),
        "schemaLinkedPropertyId": looks_like["id"],
        "name": "looks_like",
        "displayName": "looks_like",
    }


def test_package_proxies(
    audit_logger,
    configure_get,
    configure_post,
    configure_put,
    configure_delete,
    concept,
    create_concept_instance,
    create_concept,
    create_property,
    valid_user,
    api_client,
):
    get = configure_get(audit_logger)
    post = configure_post(audit_logger)
    delete = configure_delete(audit_logger)

    ci1 = create_concept_instance(concept, [])  # noqa: F841
    ci2 = create_concept_instance(concept, [])  # noqa: F841

    # Create a concept with no records:
    create_concept("medication", "Medication", "Medication")

    r = get("/v1/datasets/{dataset}/proxy/package")
    assert r.status_code == 200

    r = get("/v1/datasets/{dataset}/proxy/package/instances")
    assert r.status_code == 200
    assert r.json == []

    api_client.get_packages_response = {
        "package_id": {"content": {"nodeId": "N:package:1234", "id": 1234}}
    }

    # Create a proxy instance
    r = post(
        "/v1/datasets/{dataset}/proxy/package/instances",
        json={
            "externalId": "N:package:1234",
            "targets": [
                {
                    "direction": "FromTarget",
                    "linkTarget": {"ConceptInstance": {"id": ci1["id"]}},
                    "relationshipType": "belongs_to",
                    "relationshipData": [],
                },
                {
                    "direction": "FromTarget",
                    "linkTarget": {"ConceptInstance": {"id": ci2["id"]}},
                    "relationshipType": "has_a",
                    "relationshipData": [],
                },
            ],
        },
    )
    assert r.status_code == 201
    proxyInstance1 = r.json[0]["proxyInstance"]
    proxyInstance2 = r.json[1]["proxyInstance"]
    assert proxyInstance1["externalId"] == "N:package:1234"
    assert proxyInstance2["externalId"] == "N:package:1234"

    # Get all proxies
    r = get("/v1/datasets/{dataset}/proxy/package/instances")
    assert r.status_code == 200
    assert sorted(r.json, key=lambda p: p["id"]) == sorted(
        [proxyInstance1, proxyInstance2], key=lambda p: p["id"]
    )

    # Get a proxy directly by id
    r = get(
        "/v1/datasets/{dataset}/proxy/package/instances/{id}", id=proxyInstance1["id"]
    )
    assert r.status_code == 200
    assert r.json == proxyInstance1

    # Find the number of relations to a given package
    r = get(
        "/v1/datasets/{dataset}/proxy/package/external/N:package:1234/relationCounts"
    )
    assert r.status_code == 200
    assert r.json == [{"count": 2, "displayName": "Patient", "name": "patient"}]

    api_client.get_packages_response = {
        1234: {
            "content": {
                "nodeId": "N:package:1234",
                "id": 1234,
                "datasetNodeId": "N:dataset:7890",
                "datasetId": 7890,
            }
        }
    }

    # Check the associated records:
    # Note: for legacy reasons ("/files") is ignored from the path.

    # Non-existent models (raise a 404):
    r = get(
        "/v1/datasets/{dataset}/proxy/package/external/N:package:1234/relations/does-not-exist/files"
    )
    assert r.status_code == 404

    r = get(
        "/v1/datasets/{dataset}/proxy/package/external/N:package:1234/relations/medication/files"
    )
    assert r.status_code == 200
    assert len(r.json) == 0

    r = get(
        "/v1/datasets/{dataset}/proxy/package/external/N:package:1234/relations/patient/files"
    )
    result = r.json
    assert len(result) == 2
    assert {ci1["id"], ci2["id"]} == {r["id"] for (rr, r) in result}

    # Get files related to a concept instance
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept}/instances/{id}/files",
        concept=concept,
        id=ci1["id"],
    )
    assert r.status_code == 200
    assert len(r.json) == 1
    assert r.json[0][0]["id"] == proxyInstance1["id"]

    assert r.json[0][1] == {
        "content": {
            "id": "N:package:1234",
            "nodeId": "N:package:1234",
            "datasetId": "N:dataset:7890",
            "datasetNodeId": "N:dataset:7890",
        }
    }

    # Get files related to a concept instance (paginated)
    r = get(
        "/v1/datasets/{dataset}/concepts/{concept}/instances/{id}/files-paged?limit=10&offset=0",
        concept=concept,
        id=ci1["id"],
    )
    assert r.status_code == 200
    assert len(r.json["results"]) == 1
    assert r.json["results"][0][0] == {"id": proxyInstance1["id"]}
    assert r.json["results"][0][1] == {
        "content": {
            "id": "N:package:1234",
            "nodeId": "N:package:1234",
            "datasetId": "N:dataset:7890",
            "datasetNodeId": "N:dataset:7890",
        }
    }
    assert r.json["limit"] == 10
    assert r.json["offset"] == 0
    assert r.json["totalCount"] == 1

    # "Delete" the package from Pennsieve API
    # Both endpoints should ignore the missing package
    api_client.get_packages_response = {}

    r = get(
        "/v1/datasets/{dataset}/concepts/{concept}/instances/{id}/files",
        concept=concept,
        id=ci1["id"],
    )
    assert r.status_code == 200
    assert len(r.json) == 0

    r = get(
        "/v1/datasets/{dataset}/concepts/{concept}/instances/{id}/files-paged?limit=10&offset=0",
        concept=concept,
        id=ci1["id"],
    )
    assert r.status_code == 200
    assert len(r.json["results"]) == 0

    # Delete proxy 1
    r = delete(
        "/v1/datasets/{dataset}/proxy/package/instances/{id}", id=proxyInstance1["id"]
    )
    assert r.status_code == 200

    # Gone.
    r = get(
        "/v1/datasets/{dataset}/proxy/package/instances/{id}", id=proxyInstance1["id"]
    )
    assert r.status_code == 404

    # Relation count has decreased
    r = get(
        "/v1/datasets/{dataset}/proxy/package/external/N:package:1234/relationCounts"
    )
    assert r.status_code == 200
    assert r.json == [{"count": 1, "displayName": "Patient", "name": "patient"}]

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 11

    # Add the package response back:
    api_client.get_packages_response = {
        "package_id": {"content": {"nodeId": "N:package:1234", "id": 1234}}
    }

    # Create some new proxy instances for bulk deletion
    r = post(
        "/v1/datasets/{dataset}/proxy/package/instances",
        json={
            "externalId": "N:package:1234",
            "targets": [
                {
                    "direction": "FromTarget",
                    "linkTarget": {"ConceptInstance": {"id": ci1["id"]}},
                    "relationshipType": "belongs_to",
                    "relationshipData": [],
                },
                {
                    "direction": "FromTarget",
                    "linkTarget": {"ConceptInstance": {"id": ci2["id"]}},
                    "relationshipType": "has_a",
                    "relationshipData": [],
                },
            ],
        },
    )
    assert r.status_code == 201
    lastProxyInstance1 = r.json[0]["proxyInstance"]
    lastProxyInstance2 = r.json[1]["proxyInstance"]
    assert lastProxyInstance1["externalId"] == "N:package:1234"
    assert lastProxyInstance2["externalId"] == "N:package:1234"

    # Delete both using the bulk deletion endpoint will fail since
    # lastProxyInstance2["id"] isn't connected to ci1:
    r = delete(
        "/v1/datasets/{dataset}/proxy/package/instances/bulk",
        json={
            "sourceRecordId": ci1["id"],
            "proxyInstanceIds": [lastProxyInstance1["id"], lastProxyInstance2["id"]],
        },
    )
    assert r.status_code == 404

    # Delete the file proxy linked to ci1 should succeed:
    r = delete(
        "/v1/datasets/{dataset}/proxy/package/instances/bulk",
        json={
            "sourceRecordId": ci1["id"],
            "proxyInstanceIds": [lastProxyInstance1["id"]],
        },
    )
    assert r.status_code == 200

    # All should be gone.
    r = get("/v1/datasets/{dataset}/proxy/package/instances")
    assert r.status_code == 200
    assert len(r.json) == 1


def test_concept_instance_relation_counts(
    get,
    post,
    put,
    delete,
    valid_user,
    sample_patient_db,
    partitioned_db,
    create_linked_property,
    create_linked_property_instance,
):
    bob = sample_patient_db["records"]["bob"]
    patient = sample_patient_db["models"]["patient"]

    r = get(
        "/v1/datasets/{dataset}/concepts/{patient_id}/instances/{bob_id}/relationCounts",
        patient_id=patient.id,
        bob_id=bob.id,
    )
    assert r.status_code == 200
    assert r.json == [{"name": "visit", "displayName": "Visit", "count": 1}]

    medication = sample_patient_db["models"]["medication"]
    aspirin = sample_patient_db["records"]["aspirin"]

    # Ignore outgoing linked properties

    prefers_medication = create_linked_property(
        patient, "prefers_medication", medication
    )
    hates_medication = create_linked_property(patient, "hates_medication", medication)
    bob_prefers_aspirin = create_linked_property_instance(
        patient, bob, prefers_medication, aspirin
    )

    r = get(
        "/v1/datasets/{dataset}/concepts/{patient_id}/instances/{bob_id}/relationCounts",
        patient_id=patient.id,
        bob_id=bob.id,
    )
    assert r.status_code == 200
    assert r.json == [{"name": "visit", "displayName": "Visit", "count": 1}]

    # Ignore incoming linked properties
    r = get(
        "/v1/datasets/{dataset}/concepts/{medication_id}/instances/{aspirin_id}/relationCounts",
        patient_id=patient.id,
        aspirin_id=aspirin.id,
        medication_id=medication.id,
    )
    assert r.status_code == 200
    assert r.json == [{"name": "visit", "displayName": "Visit", "count": 2}]

    # ... unless `?includeIncomingLinkedProperties` is true.
    # (two normal relationships, one incoming linked property)
    r = get(
        "/v1/datasets/{dataset}/concepts/{medication_id}/instances/{aspirin_id}/relationCounts?includeIncomingLinkedProperties=true",
        patient_id=patient.id,
        aspirin_id=aspirin.id,
        medication_id=medication.id,
    )
    assert r.status_code == 200
    assert r.json == [
        {"name": "visit", "displayName": "Visit", "count": 2},
        {"name": "patient", "displayName": "Patient", "count": 1},
    ]

    # Count multiple relationships

    tuesday = sample_patient_db["records"]["tuesday"]
    visit = sample_patient_db["models"]["visit"]

    r = get(
        "/v1/datasets/{dataset}/concepts/{visit_id}/instances/{tuesday_id}/relationCounts",
        visit_id=visit.id,
        tuesday_id=tuesday.id,
    )
    assert r.status_code == 200
    assert sorted(r.json, key=lambda s: s["name"]) == sorted(
        [
            {"name": "patient", "displayName": "Patient", "count": 1},
            {"name": "medication", "displayName": "Medication", "count": 2},
        ],
        key=lambda s: s["name"],
    )

    # Include package proxies

    partitioned_db.create_package_proxy(
        tuesday, package_id=1234, package_node_id="N:package:1234"
    )

    r = get(
        "/v1/datasets/{dataset}/concepts/{visit_id}/instances/{tuesday_id}/relationCounts",
        visit_id=visit.id,
        tuesday_id=tuesday.id,
    )
    assert r.status_code == 200
    assert sorted(r.json, key=lambda s: s["name"]) == sorted(
        [
            {"name": "patient", "displayName": "Patient", "count": 1},
            {"name": "medication", "displayName": "Medication", "count": 2},
            {"name": "package", "displayName": "Files", "count": 1},
        ],
        key=lambda s: s["name"],
    )


def test_concept_instance_relations(
    get,
    post,
    put,
    delete,
    valid_user,
    sample_patient_db,
    create_linked_property,
    create_linked_property_instance,
):
    bob = sample_patient_db["records"]["bob"]
    tuesday = sample_patient_db["records"]["tuesday"]
    aspirin = sample_patient_db["records"]["aspirin"]
    patient = sample_patient_db["models"]["patient"]
    visit = sample_patient_db["models"]["visit"]
    medication = sample_patient_db["models"]["medication"]
    attends = sample_patient_db["relationships"]["attends"]

    r = get(
        "/v1/datasets/{dataset}/concepts/{patient_id}/instances/{bob_id}/relations/{visit_id}",
        patient_id=patient.id,
        bob_id=bob.id,
        visit_id=visit.id,
    )
    assert r.status_code == 200
    assert len(r.json) == 1
    relationship, record = r.json[0]

    assert relationship["from"] == str(bob.id)
    assert relationship["to"] == str(tuesday.id)
    assert relationship["schemaRelationshipId"] == str(attends.id)
    assert relationship["type"] == "attends"
    assert record["id"] == str(tuesday.id)
    assert record["values"][0]["dataType"] == "String"

    # Ignore outgoing linked properties

    prefers_medication = create_linked_property(
        patient, "prefers_medication", medication, display_name="Prefers Medication"
    )
    hates_medication = create_linked_property(patient, "hates_medication", medication)
    bob_prefers_aspirin = create_linked_property_instance(
        patient, bob, prefers_medication, aspirin
    )

    r = get(
        "/v1/datasets/{dataset}/concepts/{patient_id}/instances/{bob_id}/relations/{medication_id}",
        patient_id=patient.id,
        bob_id=bob.id,
        medication_id=medication.id,
    )
    assert r.status_code == 200
    assert len(r.json) == 0

    # Ignore incoming linked properties
    r = get(
        "/v1/datasets/{dataset}/concepts/{medication_id}/instances/{aspirin_id}/relations/{patient_id}",
        patient_id=patient.id,
        aspirin_id=aspirin.id,
        medication_id=medication.id,
    )
    assert r.status_code == 200
    assert len(r.json) == 0

    # ... unless the `includeIncomingLinkedProperties` query param is set
    r = get(
        "/v1/datasets/{dataset}/concepts/{medication_id}/instances/{aspirin_id}/relations/{patient_id}?includeIncomingLinkedProperties=true",
        patient_id=patient.id,
        aspirin_id=aspirin.id,
        medication_id=medication.id,
    )
    assert r.status_code == 200
    assert len(r.json) == 1

    relationship, record = r.json[0]

    assert drop_volatile(relationship) == {
        "from": str(bob.id),
        "to": str(aspirin.id),
        "schemaLinkedPropertyId": prefers_medication["id"],
        "name": "prefers_medication",
        "displayName": "Prefers Medication",
    }
    assert record["id"] == str(bob.id)


def test_create_concept_relationship_with_stub(
    post, get, create_concept, create_concept_instance
):
    patient = create_concept(
        name="patient", display_name="Patient", description="A patient"
    )
    doctor = create_concept(
        name="doctor", display_name="Doctor", description="A doctor"
    )

    patient1 = create_concept_instance(patient, [])

    doctor1 = create_concept_instance(doctor, [])

    doctor2 = create_concept_instance(doctor, [])

    r = post(
        "/v1/datasets/{dataset}/relationships",
        json={
            "name": "a_relationship",
            "displayName": "A Relationship",
            "description": "A description",
            "schema": [],
        },
    )
    assert r.status_code == 201

    # Can create a relationship directly from the stub
    r = post(
        "/v1/datasets/{dataset}/relationships/a_relationship/instances/batch",
        json=[{"from": patient1["id"], "to": doctor1["id"], "values": []}],
    )
    assert r.status_code == 200

    # Can create another relationship, without an error from duplicating the model relationship
    r = post(
        "/v1/datasets/{dataset}/relationships/a_relationship/instances/batch",
        json=[{"from": patient1["id"], "to": doctor2["id"], "values": []}],
    )
    assert r.status_code == 200


def test_concept_instance_relations_with_relationship_ordering(movie_db, get):
    _, movie_vars = movie_db

    keanu = movie_vars["Keanu"]
    movie = movie_vars["movie"]

    r = get(
        "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}?relationshipOrderBy=not-supported",
        person=str(keanu.id),
        movie=movie.name,
    )
    assert r.status_code == 415

    # in ascending order
    r = get(
        "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}?relationshipOrderBy=label",
        person=str(keanu.id),
        movie=movie.name,
    )
    assert r.status_code == 200
    assert list(unique_justseen([rr["type"] for rr, _ in r.json])) == [
        "acted_in",
        "has_leading_role",
    ]

    # descending order
    r = get(
        "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}?relationshipOrderBy=label&ascending=false",
        person=str(keanu.id),
        movie=movie.name,
    )
    assert r.status_code == 200
    assert list(unique_justseen([rr["type"] for rr, _ in r.json])) == [
        "has_leading_role",
        "acted_in",
    ]


def test_concept_instance_relations_with_default_ordering(movie_db, get):
    _, movie_vars = movie_db

    tom = movie_vars["TomC"]
    movie = movie_vars["movie"]
    r = get(
        "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}",
        person=str(tom.id),
        movie=movie.name,
    )
    assert r.status_code == 200
    assert len(r.json) == 3

    movie_titles = []
    for _, rec in r.json:
        for val in rec["values"]:
            if val["conceptTitle"]:
                movie_titles.append(val["value"])
    # the default order is the chronological order of creation fo the records
    # based on the fixtures.py movie_db, the proper order is
    # A Few Good Men, Top Gun, Jerry Maguire
    assert movie_titles == ["A Few Good Men", "Top Gun", "Jerry Maguire"]


def test_concept_instance_relations_with_record_ordering(movie_db, get):
    _, movie_vars = movie_db

    tom = movie_vars["TomC"]
    movie = movie_vars["movie"]

    # property name must exist:
    r = get(
        "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}?recordOrderBy=whatever",
        person=str(tom.id),
        movie=movie.name,
    )
    assert r.status_code == 400

    # sort by special properties "created at"/"updated at":
    for order_prop in ORDER_BY_CREATED_AT_FIELDS | ORDER_BY_UPDATED_AT_FIELDS:
        r = get(
            "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}?recordOrderBy={order_prop}",
            person=str(tom.id),
            movie=movie.name,
            order_prop=order_prop,
        )
        assert r.status_code == 200

    # sort by title - ascending
    r = get(
        "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}?recordOrderBy=title",
        person=str(tom.id),
        movie=movie.name,
    )
    assert r.status_code == 200
    values = [
        r["value"]
        for r in chain.from_iterable(rec["values"] for (_, rec) in r.json)
        if r["name"] == "title"
    ]
    assert values == ["A Few Good Men", "Jerry Maguire", "Top Gun"]

    # sort by title - decending
    r = get(
        "/v1/datasets/{dataset}/concepts/person/instances/{person}/relations/{movie}?recordOrderBy=title&ascending=false",
        person=str(tom.id),
        movie=movie.name,
    )
    assert r.status_code == 200
    values = [
        r["value"]
        for r in chain.from_iterable(rec["values"] for (_, rec) in r.json)
        if r["name"] == "title"
    ]
    assert values == ["Top Gun", "Jerry Maguire", "A Few Good Men"]


def test_concept_instance_relations_with_duplicate_values(
    sample_patient_db,
    partitioned_db,
    get,
    create_concept_instance_batch,
    create_concept_instance_relationship_batch,
):
    """
    When sorting by duplicate values, or when sorting by date with duplicate
    dates, records should still be ordered by the time they were created.
    """
    patient = sample_patient_db["models"]["patient"]
    tuesday = sample_patient_db["records"]["tuesday"]

    instances = create_concept_instance_batch(
        patient,
        [
            [{"name": "name", "value": f"Patient {i}"}, {"name": "age", "value": 25}]
            for i in range(5)
        ],
    )

    create_concept_instance_relationship_batch(
        instances, "attends", [tuesday] * len(instances)
    )

    r = get(
        "/v1/datasets/{dataset}/concepts/visit/instances/{tuesday_id}/relations/patient?recordOrderBy=createdAt",
        tuesday_id=tuesday.id,
    )
    assert r.status_code == 200

    patient_names = [
        r["value"]
        for r in chain.from_iterable(record["values"] for (_, record) in r.json)
        if r["name"] == "name"
    ]
    assert patient_names == ["Bob"] + [f"Patient {i}" for i in range(5)]

    r = get(
        "/v1/datasets/{dataset}/concepts/visit/instances/{tuesday_id}/relations/patient?recordOrderBy=age",
        tuesday_id=tuesday.id,
    )
    assert r.status_code == 200

    patient_names = [
        r["value"]
        for r in chain.from_iterable(record["values"] for (_, record) in r.json)
        if r["name"] == "name"
    ]
    assert patient_names == ["Bob"] + [f"Patient {i}" for i in range(5)]


def test_concept_instance_counts(get, post, put, delete, valid_user, sample_patient_db):
    r = get("/v1/datasets/{dataset}/concepts/patient")
    assert r.status_code == 200
    assert r.json["count"] == 2

    r = get("/v1/datasets/{dataset}/concepts/medication")
    assert r.status_code == 200
    assert r.json["count"] == 3

    r = get("/v1/datasets/{dataset}/concepts/visit")
    assert r.status_code == 200
    assert r.json["count"] == 2


def test_concept_topology(movie_db, get):

    r = get("/v1/datasets/{dataset}/concepts/does-not-exist/topology")
    assert r.status_code == 400

    r = get("/v1/datasets/{dataset}/concepts/movie/topology")
    assert r.status_code == 200
    data = r.json
    # should contain "award", "genre", and "person"
    award = [d for d in data if d["name"] == "award"][0]
    assert award["count"] == 2

    genre = [d for d in data if d["name"] == "genre"][0]
    assert genre["count"] == 6

    person = [d for d in data if d["name"] == "person"][0]
    assert person["count"] == 65


def test_graph_summary(movie_db, get):
    r = get("/v1/datasets/{dataset}/concepts/graph/summary")
    assert r.status_code == 200
    summary = r.json
    assert summary["modelCount"] == 6
    assert summary["modelRecordCount"] == 96
    assert len(summary["modelSummary"]) == 6
    assert summary["relationshipCount"] == 10
    assert summary["relationshipRecordCount"] == 132
    assert len(summary["relationshipSummary"]) == 10
    assert summary["relationshipTypeCount"] == 9
    assert len(summary["relationshipTypeSummary"]) == 9


def test_query_parsing():
    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "movie"} }, "filters": [], "joins": [], "order_by": null, "limit": null, "offset": null, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="movie"),
        filters=[],
        joins=[],
        order_by=None,
        limit=None,
        offset=None,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "movie"} }, "filters": [], "joins": [], "order_by": null, "limit": 5, "offset": 5, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="movie"),
        filters=[],
        joins=[],
        order_by=None,
        limit=5,
        offset=5,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "movie"} }, "filters": [], "joins": [], "order_by": {"Ascending": {"field": "title"} }, "limit": null, "offset": null, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="movie"),
        filters=[],
        joins=[],
        order_by=legacy.OrderBy(field="title", ascending=True),
        limit=None,
        offset=None,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "movie"} }, "filters": [{"key": "totallyBogusKey", "predicate": {"operation": "eq", "value": 4955304709811152113 } } ], "joins": [], "order_by": null, "limit": null, "offset": null, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="movie"),
        filters=[
            legacy.KeyFilter(
                key="totallyBogusKey",
                predicate=legacy.Predicate1(operation="eq", value=4955304709811152113),
            )
        ],
        joins=[],
        order_by=None,
        limit=None,
        offset=None,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "award"} }, "filters": [], "joins": [{"relationshipType": "WON", "targetType": {"concept": {"type": "genre"} }, "filters": [], "key": null } ], "order_by": null, "limit": null, "offset": null, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="award"),
        filters=[],
        joins=[
            legacy.Join(
                relationship_type="WON",
                target_type=legacy.ConceptQueryType(type="genre"),
                filters=[],
                key=None,
            )
        ],
        order_by=None,
        limit=None,
        offset=None,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "movie"} }, "filters": [], "joins": [], "order_by": null, "limit": null, "offset": null, "select": {"GroupCount": {"field": "foo", "key": null } } }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="movie"),
        filters=[],
        joins=[],
        order_by=None,
        limit=None,
        offset=None,
        select=legacy.SelectGroupCount(field="foo", key=None),
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "award"} }, "filters": [], "joins": [{"relationshipType": null, "targetType": {"concept": {"type": "genre"} }, "filters": [], "key": "joinKey"} ], "order_by": null, "limit": null, "offset": null, "select": {"GroupCount": {"field": "title", "key": null } } }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="award"),
        filters=[],
        joins=[
            legacy.Join(
                relationship_type=None,
                target_type=legacy.ConceptQueryType(type="genre"),
                filters=[],
                key="joinKey",
            )
        ],
        order_by=None,
        limit=None,
        offset=None,
        select=legacy.SelectGroupCount(field="title", key=None),
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"proxy": {"type": "package"} }, "filters": [{"key": "name", "predicate": {"operation": "neq", "value": "QLCrDvhiNM"} } ], "joins": [], "order_by": null, "limit": null, "offset": null, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ProxyQueryType(),
        filters=[
            legacy.KeyFilter(
                key="name",
                predicate=legacy.Predicate1(operation="neq", value="QLCrDvhiNM"),
            )
        ],
        joins=[],
        order_by=None,
        limit=None,
        offset=None,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"proxy": {"type": "package"} }, "filters": [{"key": "type", "predicate": {"operation": "eq", "value": "CSV"} } ], "joins": [], "order_by": null, "limit": null, "offset": null, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ProxyQueryType(),
        filters=[
            legacy.KeyFilter(
                key="type", predicate=legacy.Predicate1(operation="eq", value="CSV")
            )
        ],
        joins=[],
        order_by=None,
        limit=None,
        offset=None,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"proxy": {"type": "package"} }, "filters": [{"key": "nodeId", "predicate": {"operation": "eq", "value": "N:package:55f97c52-cc96-426a-9204-972577cce0d3"} } ], "joins": [], "order_by": null, "limit": null, "offset": null, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ProxyQueryType(),
        filters=[
            legacy.KeyFilter(
                key="nodeId",
                predicate=legacy.Predicate1(
                    operation="eq",
                    value="N:package:55f97c52-cc96-426a-9204-972577cce0d3",
                ),
            )
        ],
        joins=[],
        order_by=None,
        limit=None,
        offset=None,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type": {"concept": {"type": "movie"} }, "filters": [], "joins": [], "order_by": {"Ascending": {"field": "title"} }, "limit": 5, "offset": 5, "select": null }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="movie"),
        filters=[],
        joins=[],
        order_by=legacy.OrderBy(field="title", ascending=True),
        limit=5,
        offset=5,
        select=None,
    )

    assert legacy.GraphQuery.schema().loads(
        """{"type":{"concept":{"type":"movie"}},"filters":[],"joins":[], "order_by": { "Descending": { "field": "name" } } }"""
    ) == legacy.GraphQuery(
        type=legacy.ConceptQueryType(type="movie"),
        filters=[],
        joins=[],
        order_by=legacy.OrderBy(field="name", ascending=False),
        limit=None,
        offset=None,
        select=None,
    )


def test_legacy_querying(configure_post, audit_logger, movie_db):

    post = configure_post(audit_logger)

    # query = "find all people who won an award, but only return the 'main' type = targetValue"  (Lana & Lily Wachoski)
    payload = json.loads(
        """{"type":{"concept":{"type":"person"}},"filters":[],"joins":[{"targetType":{"concept":{"type":"award"}},"filters":[],"key":"site"}], "orderBy": { "Ascending": { "field": "name" } } }"""
    )
    # connexion converts camelcased keys in `payload` to snake case:
    r = post("/v1/datasets/{dataset}/query/run", json=payload)
    result = r.json
    assert r.status_code == 200
    assert len(result) == 2
    assert "targetValue" in result[0]
    assert result[0]["targetValue"]["type"] == "person"
    # Check sorted order:
    assert result[0]["targetValue"]["values"][0]["value"] == "Lana Wachowski"
    assert result[1]["targetValue"]["type"] == "person"
    assert result[1]["targetValue"]["values"][0]["value"] == "Lilly Wachowski"

    # Limit and offset (same as above)
    payload = json.loads(
        """{"type":{"concept":{"type":"person"}},"filters":[],"joins":[{"targetType":{"concept":{"type":"award"}},"filters":[],"key":"site"}],"orderBy": { "Descending": { "field": "born" } }, "limit": 1 }"""
    )
    r = post("/v1/datasets/{dataset}/query/run", json=payload)
    result = r.json
    assert r.status_code == 200
    assert len(result) == 1
    payload = json.loads(
        """{"type":{"concept":{"type":"person"}},"filters":[],"joins":[{"targetType":{"concept":{"type":"award"}},"filters":[],"key":"site"}],"orderBy": { "Descending": { "field": "born" } }, "limit": 1, "offset": 1 }"""
    )
    r = post("/v1/datasets/{dataset}/query/run", json=payload)
    result = r.json
    assert r.status_code == 200
    assert len(result) == 1

    # same as the query above, but include the award as well:
    payload = json.loads(
        """{"type":{"concept":{"type":"person"}},"filters":[],"joins":[{"targetType":{"concept":{"type":"award"}},"filters":[],"key":"award"}],"orderBy": { "Descending": { "field": "born" } }, "offset":0,"limit":50,"select":{"Concepts":{"join_keys":["award"]}}}"""
    )
    r = post("/v1/datasets/{dataset}/query/run", json=payload)
    result = r.json
    assert r.status_code == 200
    assert len(result) == 2
    assert "targetValue" in result[0]
    assert result[0]["targetValue"]["type"] == "person"
    assert result[0]["targetValue"]["values"][0]["value"] == "Lilly Wachowski"
    assert result[0]["award"]["values"][0]["value"] == "Academy Award"
    assert result[1]["targetValue"]["type"] == "person"
    assert result[1]["targetValue"]["values"][0]["value"] == "Lana Wachowski"
    assert result[1]["award"]["values"][0]["value"] == "Academy Award"

    # Count group results by person:
    payload = json.loads(
        """{"type":{"concept":{"type":"person"}},"filters":[],"joins":[{"targetType":{"concept":{"type":"award"}},"filters":[],"key":"award"}],"orderBy": { "Descending": { "field": "born" } }, "offset":0,"limit":50,"select":{"GroupCount": {"field": "name", "key": "person"}}}"""
    )
    r = post("/v1/datasets/{dataset}/query/run", json=payload)
    result = r.json
    assert r.status_code == 200
    assert result == [{"Lana Wachowski": 1, "Lilly Wachowski": 1}]

    # Count group results by award:
    payload = json.loads(
        """{"type":{"concept":{"type":"person"}},"filters":[],"joins":[{"targetType":{"concept":{"type":"award"}},"filters":[],"key":"award"}],"orderBy": { "Descending": { "field": "born" } }, "offset":0,"limit":50,"select":{"GroupCount": {"field": "name", "key": "award"}}}"""
    )
    r = post("/v1/datasets/{dataset}/query/run", json=payload)
    result = r.json
    assert r.status_code == 200
    assert result == [{"Academy Award": 2}]

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 6


def test_legacy_datatype_conversions():

    assert common.to_legacy_data_type(dt.Array(items=dt.String())) == {
        "type": "array",
        "items": {"type": "String", "format": None},
    }
    assert common.to_legacy_data_type(
        dt.Enumeration(items=dt.String(), enum=["red", "green"])
    ) == {
        "type": "enum",
        "items": {"type": "String", "format": None, "enum": ["red", "green"]},
    }


def test_get_property_unit_options(get):
    r = get("/v1/datasets/{dataset}/properties/units")
    assert r.status_code == 200
    assert len(r.json) == 17


def test_get_property_string_subtype_options(get):
    r = get("/v1/datasets/{dataset}/properties/strings")
    assert r.status_code == 200
    assert r.json == dt.STRING_SUBTYPES


def test_v1_api_is_also_mounted_at_service_root(get, post):
    r = post(
        "/datasets/{dataset}/concepts",
        json={"name": "patient", "displayName": "Patient", "description": "Test model"},
    )
    assert r.status_code == 201

    r = get("/datasets/{dataset}/concepts")
    assert r.status_code == 200
    assert len(r.json) == 1
