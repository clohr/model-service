import csv
import io
import json
import logging
import uuid
from collections import OrderedDict
from copy import copy
from dataclasses import replace
from uuid import UUID

import pytest
from flask import current_app

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
from core.dtos import api
from server.errors import InvalidDatasetError
from server.models import DatasetId, PackageId, PackageNodeId
from server.models import datatypes as dt
from server.models.query import Operator as op

log = logging.getLogger(__file__)


def get_property(data, key):
    """
    Get a property, either from a JSON response dictionary or a dataclass.
    """
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


@pytest.fixture(scope="function")
def model_configure(
    client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    default_organization_id, _ = valid_organization
    default_dataset_id, _ = valid_dataset

    def build(
        name="patient",
        displayName=None,
        description=None,
        organization_id=None,
        dataset_id=None,
    ):
        if displayName is None:
            displayName = name.capitalize()
        if description is None:
            description = f"{name.capitalize()} model"

        org = organization_id or default_organization_id.id
        ds = dataset_id or default_dataset_id.id

        r = client.post(
            f"/v2/organizations/{org}/datasets/{ds}/models",
            headers=dict(**auth_headers, **trace_id_headers),
            json={"name": name, "displayName": displayName, "description": description},
        )
        assert r.status_code == 201
        return r.json

    return build


@pytest.fixture(scope="function")
def model(
    client,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    name="patient",
    displayName="Patient",
    description="Patient model",
    organization_id=None,
    dataset_id=None,
):
    default_organization_id, _ = valid_organization
    default_dataset_id, _ = valid_dataset

    org = organization_id or default_organization_id.id
    ds = dataset_id or default_dataset_id.id

    r = client.post(
        f"/v2/organizations/{org}/datasets/{ds}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"name": name, "displayName": displayName, "description": description},
    )

    if r.status_code > 299:
        log.warning(f"model: {r.json}")
    assert r.status_code == 201

    return r.json


@pytest.fixture(scope="function")
def create_model_relationship(
    client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    default_organization_id, _ = valid_organization
    default_dataset_id, _ = valid_dataset

    def build(
        model_, rel_type, n, one_to_many=False, organization_id=None, dataset_id=None
    ):

        org = organization_id or default_organization_id.id
        ds = dataset_id or default_dataset_id.id

        r = client.post(
            f"/v2/organizations/{org}/datasets/{ds}/models/{get_property(model_, 'id')}/relationships",
            headers=dict(**auth_headers, **trace_id_headers),
            json={"type": rel_type, "to": n["id"], "oneToMany": one_to_many},
        )

        if r.status_code > 299:
            log.warning(f"create_model_relationship: {r.json}")
        assert r.status_code == 201

        return r.json

    return build


@pytest.fixture(scope="function")
def create_property(
    client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    default_organization_id, _ = valid_organization
    default_dataset_id, _ = valid_dataset

    def build(
        model_id,
        name="name",
        data_type=dt.String(),
        default_value=None,
        model_title=False,
        organization_id=None,
        dataset_id=None,
    ):
        data = {
            "name": name,
            "displayName": name.title(),
            "description": "All 'bout it",
            "dataType": data_type.to_dict(),
            "modelTitle": model_title,
        }

        org = organization_id or default_organization_id.id
        ds = dataset_id or default_dataset_id.id

        if default_value is not None:
            data["default"] = True
            data["defaultValue"] = default_value
        else:
            data["default"] = False
            data["defaultValue"] = None

        r = client.put(
            f"/v2/organizations/{org}/datasets/{ds}/models/{model_id}/properties",
            headers=dict(**auth_headers, **trace_id_headers),
            json=[data],
        )

        if r.status_code > 299:
            log.warning(f"create_property: {r.json}")
            log.error("FAILURE: ", r.get_data())
        assert r.status_code == 200

        return r.json

    return build


@pytest.fixture(scope="function")
def create_record(
    client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    default_organization_id, _ = valid_organization
    default_dataset_id, _ = valid_dataset

    def build(model_, values, organization_id=None, dataset_id=None):

        org = organization_id or default_organization_id.id
        ds = dataset_id or default_dataset_id.id

        r = client.post(
            f"/v2/organizations/{org}/datasets/{ds}/models/{model_['id']}/records",
            headers=dict(**auth_headers, **trace_id_headers),
            json={"values": values},
        )

        if r.status_code > 299:
            log.warning(f"create_record: {r.json}")
        assert r.status_code == 201

        return r.json

    return build


def create_relationship(client, headers, organization, dataset, record, **json):
    r = client.post(
        f"/v2/organizations/{organization}/datasets/{dataset}/records/{record['id']}/relationships",
        headers=headers,
        json=json,
    )

    if r.status_code > 299:
        log.error(f"create_relationship: {r.json}")
    assert r.status_code == 201

    return r.json


def get_relationship(client, headers, organization, dataset, record, relationship):
    r = client.get(
        f"/v2/organizations/{organization}/datasets/{dataset}/records/{record['id']}/relationships/{relationship['id']}",
        headers=headers,
    )

    if r.status_code > 299:
        log.warning(f"get_relationship: {r.json}")
    assert r.status_code == 200

    return r.json


def delete_relationship(client, headers, organization, dataset, record, relationship):
    r = client.delete(
        f"/v2/organizations/{organization}/datasets/{dataset}/records/{record['id']}/relationships/{relationship['id']}",
        headers=headers,
    )

    if r.status_code > 299:
        log.warning(f"delete_relationship: {r.json}")
    assert r.status_code == 200


def test_models(
    client,
    jobs_client,
    auth_headers,
    trace_id_headers,
    trace_id,
    valid_organization,
    valid_dataset,
    valid_user,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"name": "patient", "displayName": "Patient", "description": "Test model"},
    )
    assert r.status_code == 201
    created = r.json

    # "CreateModel" should be emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateModel(id=created["id"], name=created["name"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Creating models with a reserved name should fail:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "name": "file",
            "displayName": "File",
            "description": "Another test model",
        },
    )
    # Make sure the changelog event is emitted:
    assert r.status_code == 400

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == [created]

    model_id = created["id"]
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model_id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == created

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model_id}",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"name": "doctor", "displayName": "Doctor", "description": "Test model 2"},
    )

    assert r.status_code == 200
    assert r.json["name"] == "doctor"
    assert r.json["displayName"] == "Doctor"
    assert r.json["description"] == "Test model 2"

    # "UpdateModel" should be emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=UpdateModel(id=r.json["id"], name=r.json["name"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model_id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # "DeleteModel" should be emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=DeleteModel(id=model_id, name="doctor"),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model_id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 404


def test_models_rejecting_invalid_names(
    client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    # Test name validation:
    bad_names = [
        " \n  \r\t   ",
        "$name",
        "@name",
        "3leading_number",
        "111",
        "foo-baz",
        "toolong" * 100,
    ]
    for name in bad_names:
        r = client.post(
            f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
            headers=dict(**auth_headers, **trace_id_headers),
            json={"name": name, "displayName": name, "description": "Test model"},
        )
        assert r.status_code == 400
        msg = r.json["message"].lower()
        assert (
            "validation error" in msg or "name empty" in msg or "name too long" in msg
        )

    # Create a good one:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"name": "patient", "displayName": "Patient", "description": "Test model"},
    )
    assert r.status_code == 201
    created = r.json
    model_id = created["id"]

    # Attempt to update with a bad name:
    for name in bad_names:
        r = client.put(
            f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model_id}",
            headers=dict(**auth_headers, **trace_id_headers),
            json={"name": name, "displayName": name, "description": "Test model 2"},
        )
        assert r.status_code == 400


def test_get_model_returns_not_found_when_model_does_not_exist(
    client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{uuid.uuid4()}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 404


def test_disallow_duplicate_model_names_in_same_dataset(
    client,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model_configure,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"name": "model_1", "displayName": "Model #1", "description": ""},
    )
    assert r.status_code == 201

    # Cannot create a new model with the same name
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "name": "model_1",
            "displayName": "The same name as model-1",
            "description": "",
        },
    )
    assert r.status_code == 400

    # Create a new model with a unique name:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"name": "model_2", "displayName": "A unique name", "description": ""},
    )
    assert r.status_code == 201
    model2 = r.json["id"]

    # Can't change the name of a new model (when selected by ID) to that of
    # an old model:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model2}",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "name": "model_1",
            "displayName": "The same name as model-1",
            "description": "",
        },
    )
    assert r.status_code == 400

    # Can update other properties of a model, keeping the same name
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model2}",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "name": "model_2",
            "displayName": "A unique name",
            "description": "A new description for this same model",
        },
    )
    assert r.status_code == 200
    assert r.json["name"] == "model_2"
    assert r.json["displayName"] == "A unique name"
    assert r.json["description"] == "A new description for this same model"


def test_allow_duplicate_model_names_in_different_datasets(
    client,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    other_valid_dataset,
    model_configure,
):
    organization_id, _ = valid_organization
    # Create in the primary dataset:
    primary_dataset_id, _ = valid_dataset
    secondary_dataset_id, _ = other_valid_dataset

    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{primary_dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"name": "model_1", "displayName": "Model #1", "description": "Model #1"},
    )
    assert r.status_code == 201

    # Create in the secondary dataset:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{secondary_dataset_id.id}/models",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "name": "model_1",
            "displayName": "A unique name",
            "description": "Totally new Model #2 description",
        },
    )
    assert r.status_code == 201


def test_disallow_model_deletion_when_records_exist(
    client,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model,
    create_record,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    _ = create_record(model, {})

    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 400


def test_lookup_model_by_name(
    client,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model_configure,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    model_configure("patient", "Patient", "Patient")
    model_configure("medication", "Medication", "Medication")
    model_configure("doctor", "Doctor", "Doctor")

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/patient",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/doesnotexist",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 404


# An example of a wild property name with many different types of typically
# prohibited characters.
EXTREME_PROPERTY_NAME: str = "power_level/(sμper.power*level * Σ + brain-power-level=!)"

REASONABLE_PROPERTY_NAME: str = "power_level"


def test_model_properties(
    client, auth_headers, trace_id_headers, valid_organization, valid_dataset, model
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    # Create a property (fail)
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": EXTREME_PROPERTY_NAME,
                "displayName": "Power level",
                "description": "Off the charts",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 400
    # Create a property (OK)
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": REASONABLE_PROPERTY_NAME,
                "displayName": "Power level",
                "description": "Off the charts",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200

    # Can get the property
    # TODO: deserialize to List[ModelProperty]
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # Re-get by name:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['name']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert len(r.json) == 1
    prop = r.json[0]
    assert prop["name"] == REASONABLE_PROPERTY_NAME
    assert prop["displayName"] == "Power level"
    assert prop["dataType"] == dt.String().to_dict()

    # Delete the property
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties/{prop['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # The property no longer exists
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert len(r.json) == 0

    # Create a property with a string subtype
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": REASONABLE_PROPERTY_NAME,
                "displayName": "Power level",
                "description": "Off the charts",
                "dataType": {"type": "String", "format": "Email"},
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200

    # Create a property with a Date subtype
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "some_other_property_name",
                "displayName": "some other display name",
                "description": "Off the charts",
                "dataType": {"type": "String", "format": "Date"},
                "modelTitle": False,
            }
        ],
    )
    assert r.status_code == 200

    # Cannot create a property with an invalid string subtype
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "yet_another_name",
                "displayName": "yet another display name",
                "description": "Off the charts",
                "dataType": {"type": "String", "format": "foo"},
                "modelTitle": False,
            }
        ],
    )
    assert r.status_code == 400


def test_update_properties(
    client,
    auth_headers,
    trace_id_headers,
    trace_id,
    jobs_client,
    model,
    valid_organization,
    valid_dataset,
    valid_user,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    # Create a property:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": REASONABLE_PROPERTY_NAME,
                "displayName": "Power level",
                "description": "not good",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200

    # Model property should be marked as updated:
    jobs_client.send_changelog_events.assert_called_with(
        dataset_id=dataset_id.id,
        events=[
            CreateModelProperty(
                property_name=REASONABLE_PROPERTY_NAME,
                model_id=UUID(model["id"]),
                model_name=model["name"],
            )
        ],
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Get properties:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['name']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    ps = r.json
    assert len(ps) == 1

    # Update the property:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "id": ps[0]["id"],
                "name": ps[0]["name"],
                "dataType": ps[0]["dataType"],
                "displayName": ps[0]["displayName"],
                "description": "Much better",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['name']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    ps = r.json
    assert len(ps) == 1
    assert ps[0]["description"] == "Much better"

    # Model property should be marked as updated:
    jobs_client.send_changelog_events.assert_called_with(
        dataset_id=dataset_id.id,
        events=[
            UpdateModelProperty(
                property_name=REASONABLE_PROPERTY_NAME,
                model_id=UUID(model["id"]),
                model_name=model["name"],
            )
        ],
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Try to change an immutable property (name + property):
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "id": ps[0]["id"],
                "name": "new_name",
                "dataType": ps[0]["dataType"],
                "displayName": ps[0]["displayName"],
                "description": "Much better",
                "modelTitle": True,
            }
        ],
    )
    # Name changed:
    assert r.status_code == 400
    assert "cannot change immutable property" in r.json["message"].lower()

    # Try to change an immutable property (dataType):
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "id": ps[0]["id"],
                "name": ps[0]["name"],
                "dataType": "Long",
                "displayName": ps[0]["displayName"],
                "description": "Much better",
                "modelTitle": True,
            }
        ],
    )
    # Datatype changed:
    assert r.status_code == 400
    assert "cannot change immutable property" in r.json["message"].lower()


def test_update_properties_with_scientific_units(
    client, auth_headers, trace_id_headers, model, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    # Create a numeric property with a scientific unit:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": REASONABLE_PROPERTY_NAME,
                "displayName": "Power level",
                "description": "not good",
                "dataType": {"type": "Long", "unit": "s"},
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200

    # Get properties:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['name']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    ps = r.json
    assert len(ps) == 1

    # Update the property's scientific unit:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "id": ps[0]["id"],
                "name": ps[0]["name"],
                "dataType": {"type": "Long", "unit": "ms"},
                "displayName": ps[0]["displayName"],
                "description": "not good",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200
    assert r.json[0]["dataType"]["unit"] == "ms"

    # fail to update the property's type:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "id": ps[0]["id"],
                "name": ps[0]["name"],
                "dataType": {"type": "Double", "unit": "ms"},
                "displayName": ps[0]["displayName"],
                "description": "not good",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 400

    # Create a numeric array property with a scientific unit:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "a_different_less_extreme_name",
                "displayName": "Another Power level",
                "description": "not good",
                "dataType": {"type": "array", "items": {"type": "Long", "unit": "s"}},
                "modelTitle": False,
            }
        ],
    )

    assert r.status_code == 200
    id = r.json[0]["id"]

    # Get new property:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['name']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    p = list(filter(lambda x: x["id"] == id, r.json))[0]

    # Update the property's scientific unit:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "id": p["id"],
                "name": p["name"],
                "dataType": {"type": "array", "items": {"type": "Long", "unit": "ms"}},
                "displayName": p["displayName"],
                "description": "not good",
                "modelTitle": False,
            }
        ],
    )

    assert r.status_code == 200
    assert r.json[0]["dataType"]["items"]["unit"] == "ms"

    # fail to update the property's type:
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "id": p["id"],
                "name": p["name"],
                "dataType": {"type": "Long", "unit": "ms"},
                "displayName": p["displayName"],
                "description": "not good",
                "modelTitle": False,
            }
        ],
    )
    assert r.status_code == 400


def test_delete_properties(
    configure_client,
    audit_logger,
    jobs_client,
    auth_headers,
    trace_id_headers,
    trace_id,
    model,
    valid_organization,
    valid_dataset,
    valid_user,
):
    client = configure_client(audit_logger)

    with client.application.app_context():
        conf = current_app.config["config"]
        current_app.config["config"] = replace(
            conf, max_record_count_for_property_deletion=2
        )

    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "foo",
                "displayName": "Foo",
                "description": "Foo",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200
    props = r.json

    # Create a record:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {"foo": "bar"}},  # strings as numbers should fail
    )
    assert r.status_code == 201
    record = r.json

    # Deleting the property should fail:
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties/{props[0]['name']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 422
    assert "violation: model property in use" in r.json["message"].lower()
    assert 1 == r.json["usageCount"]

    # Delete the record:
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # Deleting the property should be successful:
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties/{props[0]['name']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # Model should be marked as updated:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=DeleteModelProperty(
            property_name=props[0]["name"],
            model_id=UUID(model["id"]),
            model_name=model["name"],
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # create the property again
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "foo",
                "displayName": "Foo",
                "description": "Foo",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200
    props = r.json

    # Create a record again:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {"foo": "bar"}},  # strings as numbers should fail
    )
    assert r.status_code == 201
    record = r.json

    # Attempting to delete a non existent property (or the @id property) should fail
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties/@id?modifyRecords=true",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 404

    # Deleting the property should be successful if we specify that we want to modify records:
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties/{props[0]['name']}?modifyRecords=true",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # NOTE: if we remove the only property from a record, it ends up orphaned (no values)!
    # Delete the record so it doesn't mess with subsequent tests:
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # create two properties on the model
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "foo",
                "displayName": "Foo",
                "description": "Foo",
                "dataType": "String",
                "modelTitle": False,
            },
            {
                "name": "bar",
                "displayName": "Bar",
                "description": "Bar",
                "dataType": "String",
                "modelTitle": True,
            },
        ],
    )
    assert r.status_code == 200
    props = r.json

    # Create 3 records (max for deletion set to 2 in the tests), only 2 of which have a "foo" property
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records/batch",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {"values": {"bar": "bar", "foo": "bar"}},
            {"values": {"bar": "bar", "foo": "baz"}},
            {"values": {"bar": "bar"}},
        ],
    )
    assert r.status_code == 201

    # Deleting the property should succeed because while the max record count for deletion is set to 2 in the tests,
    # only 2 records actually have the foo property specified
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties/{props[0]['name']}?modifyRecords=true",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # make sure the bar properties remain on the records
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
    )

    for record in r.json["results"]:
        assert "foo" not in record["values"]
        assert record["values"]["bar"] is not None

    # create the property again
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "foo",
                "displayName": "Foo",
                "description": "Foo",
                "dataType": "String",
                "modelTitle": False,
            }
        ],
    )
    assert r.status_code == 200
    props = r.json

    # Create 3 records (max for deletion set to 2 in the tests)
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records/batch",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {"values": {"bar": "1", "foo": "bar"}},
            {"values": {"bar": "2", "foo": "baz"}},
            {"values": {"bar": "3", "foo": "qux"}},
        ],
    )
    assert r.status_code == 201

    # Deleting the property should fail because the max record count for deletion is set to 2 in the tests
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties/{props[0]['name']}?modifyRecords=true",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 400
    assert (
        r.json["detail"]
        == "Cannot delete properties that are used on > 2 records. This property is used on 3"
    )

    # Make sure we've emitting logging messages:
    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 1


def test_reject_invalid_model_property_name(
    client, auth_headers, trace_id_headers, model, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    bad_names = [
        " \n  \r\t   ",
        "$name",
        "@name",
        "3leading_number",
        "111",
        "foo-baz",
        "toolong" * 100,
    ]

    for name in bad_names:
        r = client.put(
            f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
            headers=dict(**auth_headers, **trace_id_headers),
            json=[
                {
                    "name": name,
                    "displayName": "Name",
                    "description": "Will it break?",
                    "dataType": "String",
                    "modelTitle": True,
                }
            ],
        )
        assert r.status_code == 400
        msg = r.json["message"].lower()
        assert (
            "validation error" in msg or "name empty" in msg or "name too long" in msg
        )


def test_reject_multiple_same_property_names(
    client, auth_headers, trace_id_headers, model, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "foo",
                "displayName": "Foo",
                "description": "Foo",
                "dataType": "String",
                "modelTitle": True,
            },
            {
                "name": "bar",
                "displayName": "Bar",
                "description": "Bar",
                "dataType": "String",
            },
            {
                "name": "baz",
                "displayName": "Baz",
                "description": "Baz",
                "dataType": "String",
            },
            {
                "name": "foo",
                "displayName": "Quux",
                "description": "Quux",
                "dataType": "String",
            },
        ],
    )
    assert r.status_code == 400
    assert "multiple properties with name" in r.json["message"].lower()


def test_reject_multiple_model_titles(
    client, auth_headers, trace_id_headers, model, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "foo",
                "displayName": "Foo",
                "description": "Foo",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 200

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "bar",
                "displayName": "Bar",
                "description": "Bar",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )
    assert r.status_code == 400
    assert "violation: only 1 model title property allowed" in r.json["message"].lower()


def test_bad_records_request_invalid_types(
    client, auth_headers, trace_id_headers, model, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "bar",
                "displayName": "Bar",
                "description": "Bar",
                "dataType": "String",
                "modelTitle": True,
                "default": True,
                "defaultValue": 99,
            }
        ],
    )
    assert r.status_code == 400
    assert "invalid default value" in r.json["message"].lower()


def test_mistyped_default_property_value_fails(
    client,
    auth_headers,
    trace_id_headers,
    model,
    valid_organization,
    valid_dataset,
    create_property,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            },
            {
                "name": "age",
                "displayName": "Age",
                "description": "",
                "dataType": "Long",
            },
        ],
    )

    # Create a new record
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "values": {"name": "Alice", "age": "10"}  # strings as numbers should fail
        },
    )
    assert r.status_code == 400
    assert r.json["errors"][0]["name"] == "age"


def test_default_property_values(
    client,
    auth_headers,
    trace_id_headers,
    model,
    valid_organization,
    valid_dataset,
    create_property,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            },
            {
                "name": "age",
                "displayName": "Age",
                "description": "",
                "dataType": "Long",
                "default": True,
                "defaultValue": 99,
            },
            {
                "name": REASONABLE_PROPERTY_NAME,
                "displayName": "Power Level",
                "description": "",
                "dataType": "String",
                "default": True,
                "defaultValue": "OK",
            },
            {
                "name": "upper_limit",
                "displayName": "Upper limit",
                "description": "Upper limit",
                "dataType": "String",
                "required": False,
            },
        ],
    )

    # Create a new record
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records/batch",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {"values": {"name": "Alice"}},
            {"values": {"name": "Bob", "age": 40, "upper_limit": "VERY"}},
            {"values": {"name": "Carl", REASONABLE_PROPERTY_NAME: "Out of control"}},
        ],
    )
    assert r.status_code == 201
    r1, r2, r3 = r.json

    assert (
        r1["values"]["name"] == "Alice"
        and r1["values"]["age"] == 99
        and r1["values"][REASONABLE_PROPERTY_NAME] == "OK"
        and r1["values"]["upper_limit"] is None
    )
    assert (
        r2["values"]["name"] == "Bob"
        and r2["values"]["age"] == 40
        and r2["values"][REASONABLE_PROPERTY_NAME] == "OK"
        and r2["values"]["upper_limit"] == "VERY"
    )
    assert (
        r3["values"]["name"] == "Carl"
        and r3["values"]["age"] == 99
        and r3["values"][REASONABLE_PROPERTY_NAME] == "Out of control"
        and r3["values"]["upper_limit"] is None
    )


def test_model_relationships(
    client,
    auth_headers,
    trace_id_headers,
    model_configure,
    valid_organization,
    valid_dataset,
    valid_user,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    patient = model_configure("patient")
    doctor = model_configure("doctor")

    # Create a new relationship (by name of the doctor model):
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['name']}/relationships",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"type": "treats", "to": patient["id"], "oneToMany": False},
    )

    assert r.status_code == 201
    res = r.json

    assert res["type"] == "TREATS"
    assert res["from"] == doctor["id"]
    assert res["to"] == patient["id"]
    relationship_id = res["id"]

    # Get the relationship (by model name):
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['name']}/relationships/{relationship_id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    res = r.json
    assert res["id"] == relationship_id
    assert res["type"] == "TREATS"
    assert res["from"] == doctor["id"]
    assert res["to"] == patient["id"]
    assert "createdAt" in res
    assert "createdBy" in res and res["createdBy"] == user_node_id
    assert "updatedAt" in res
    assert "updatedBy" in res and res["updatedBy"] == user_node_id

    # Update it (by model ID):
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/relationships/{relationship_id}",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"displayName": "Treated By"},
    )
    assert r.status_code == 200
    assert r.json["displayName"] == "Treated By"

    # Get all relationships (by model name):
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['name']}/relationships",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    res = r.json
    assert len(res) == 1
    assert res[0]["id"] == relationship_id

    # Delete it
    client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/relationships/{relationship_id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )

    # No more relationships:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/relationships",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert len(r.json) == 0

    # and the specific relationship shouldn't be found:
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/relationships/{relationship_id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 404


def test_delete_model_relationship_constraints(
    client,
    auth_headers,
    trace_id_headers,
    model_configure,
    valid_organization,
    valid_dataset,
    create_property,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    patient = model_configure("patient")
    doctor = model_configure("doctor")

    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{patient['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )

    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )

    # Create a new relationship:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/relationships",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"type": "treats", "to": patient["id"], "oneToMany": False},
    )
    assert r.status_code == 201
    treats = r.json

    # Create a patient and doctor record:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{patient['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {"name": "Homer"}},
    )
    assert r.status_code == 201
    p = r.json

    # Create a patient and doctor record:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {"name": "Dr. Nick"}},
    )
    assert r.status_code == 201
    d = r.json

    # Create a record relationship:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{d['id']}/relationships",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"type": "treats", "to": p["id"]},
    )
    assert r.status_code == 201
    relationship = r.json

    # Check it exists:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{d['id']}/relationships/{relationship['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    res = r.json
    assert relationship["id"] == res["id"]

    # Try to delete the corresponding model relationship. It should fail:
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/relationships/{treats['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 400

    # Make sure the record relationship still exists:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{d['id']}/relationships/{relationship['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    res = r.json
    assert relationship["id"] == res["id"]


def test_records(
    configure_client,
    audit_logger,
    jobs_client,
    auth_headers,
    trace_id_headers,
    trace_id,
    model,
    valid_organization,
    valid_dataset,
    valid_user,
    create_property,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    client = configure_client(audit_logger)

    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            },
            {
                "name": "age",
                "displayName": "Age",
                "description": "",
                "dataType": "Long",
            },
            {
                "name": REASONABLE_PROPERTY_NAME,
                "displayName": "Power level",
                "description": "Power level",
                "dataType": "String",
            },
            {
                "name": "date_of_birth",
                "displayName": "Date of birth",
                "description": "Power level",
                "dataType": "Date",
            },
        ],
    )
    assert r.status_code == 200

    # Get all records - none exist yet
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json["results"] == []

    # Create a new record
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "values": {
                "name": "Alice",
                "age": 10,
                REASONABLE_PROPERTY_NAME: "Pretty good",
                "date_of_birth": "2009-11-14",
            }
        },
    )
    assert r.status_code == 201
    record = r.json
    values = record["values"]
    assert len(values) == 4
    assert values["name"] == "Alice"
    assert values["age"] == 10
    assert values[REASONABLE_PROPERTY_NAME] == "Pretty good"
    assert values["date_of_birth"] == "2009-11-14T00:00:00+00:00"

    # Make sure "CreateRecord" was emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(id=UUID(record["id"]), name="Alice", model_id=model["id"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Get the record
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == record

    # Get all records
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    rjson = r.json
    # Remove the cursor info:
    rjson.pop("nextPage", None)
    assert rjson["results"] == [record]

    # Update the values of the record
    r = client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "values": {
                "name": "Bob",
                "age": 9,
                REASONABLE_PROPERTY_NAME: "Just OK",
                "date_of_birth": "2008-11-14",
            }
        },
    )
    assert r.status_code == 200
    record = r.json
    values = record["values"]
    assert len(values) == 4
    assert values["name"] == "Bob"
    assert values["age"] == 9
    assert values[REASONABLE_PROPERTY_NAME] == "Just OK"
    assert values["date_of_birth"] == "2008-11-14T00:00:00+00:00"

    # Emit UpdateRecord
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=UpdateRecord(
            id=UUID(record["id"]),
            name="Alice",
            model_id=model["id"],
            properties=[
                UpdateRecord.PropertyDiff(
                    name="name",
                    data_type=dt.String(),
                    old_value="Alice",
                    new_value="Bob",
                ),
                UpdateRecord.PropertyDiff(
                    name="age", data_type=dt.Long(), old_value=10, new_value=9
                ),
                UpdateRecord.PropertyDiff(
                    name=REASONABLE_PROPERTY_NAME,
                    data_type=dt.String(),
                    old_value="Pretty good",
                    new_value="Just OK",
                ),
                UpdateRecord.PropertyDiff(
                    name="date_of_birth",
                    data_type=dt.Date(),
                    old_value="2009-11-14T00:00:00+00:00",
                    new_value="2008-11-14T00:00:00+00:00",
                ),
            ],
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Delete the record
    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # Emit DeleteRecord
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=DeleteRecord(id=UUID(record["id"]), name="Bob", model_id=model["id"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # No longer visible
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 404

    # Make sure the audit log emits messages:
    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 2


def test_create_record_with_no_properties(
    client, auth_headers, trace_id_headers, model, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    # by default `model` has no properties
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {}},
    )
    assert r.status_code == 201
    result = r.json
    assert "id" in result
    assert len(result["values"]) == 0


def test_cannot_create_record_without_model_title_property(
    client, auth_headers, trace_id_headers, model, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            },
            {
                "name": "age",
                "displayName": "Age",
                "description": "",
                "dataType": "Long",
                "required": False,
            },
            {
                "name": "description",
                "displayName": "Description",
                "description": "Description",
                "dataType": "String",
                "required": True,
            },
        ],
    )
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {"age": 14, "description": "interesting description"}},
    )
    assert r.status_code == 400


def test_record_batch(
    client,
    jobs_client,
    auth_headers,
    trace_id_headers,
    trace_id,
    model,
    valid_organization,
    valid_dataset,
    valid_user,
    create_property,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            },
            {
                "name": "age",
                "displayName": "Age",
                "description": "",
                "dataType": "Long",
            },
            {
                "name": REASONABLE_PROPERTY_NAME,
                "displayName": "Power level",
                "description": "Power level",
                "dataType": "String",
            },
            {
                "name": "description",
                "displayName": "Description",
                "description": "Description",
                "dataType": "String",
                "required": False,
            },
            # Required
            {
                "name": "required_value",
                "displayName": "Required value",
                "description": "Required value",
                "dataType": "String",
                "required": True,
            },
        ],
    )

    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records/batch",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "values": {
                    "name": "Alice",
                    "age": 10,
                    REASONABLE_PROPERTY_NAME: "Exceptional",
                }
            },
            # Missing "required_value"
            {
                "values": {
                    "name": "Bob",
                    "age": 9,
                    REASONABLE_PROPERTY_NAME: "Not great",
                }
            },
        ],
    )
    assert r.status_code == 400

    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{model['id']}/records/batch",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "values": {
                    "name": "Alice",
                    "age": 10,
                    REASONABLE_PROPERTY_NAME: "Exceptional",
                    "description": "It's here",
                    "required_value": "huh",
                }
            },
            # Missing "required_value"
            {
                "values": {
                    "name": "Bob",
                    "age": 9,
                    REASONABLE_PROPERTY_NAME: "Not great",
                    "required_value": "hmm",
                }
            },
        ],
    )
    assert r.status_code == 201

    assert len(r.json) == 2
    alice, bob = r.json
    assert (
        alice["values"]["name"] == "Alice"
        and alice["values"]["age"] == 10
        and alice["values"][REASONABLE_PROPERTY_NAME] == "Exceptional"
        and alice["values"]["description"] == "It's here"  # Optional, provided
        and alice["values"]["required_value"] == "huh"
    )
    assert (
        bob["values"]["name"] == "Bob"
        and bob["values"]["age"] == 9
        and bob["values"][REASONABLE_PROPERTY_NAME] == "Not great"
        and bob["values"]["description"] is None  # Optional, omitted
        and bob["values"]["required_value"] == "hmm"
    )

    jobs_client.send_changelog_events.assert_called_with(
        dataset_id=dataset_id.id,
        events=[
            CreateRecord(id=UUID(alice["id"]), name="Alice", model_id=model["id"]),
            CreateRecord(id=UUID(bob["id"]), name="Bob", model_id=model["id"]),
        ],
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )


def test_record_relationships(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    trace_id,
    jobs_client,
    model_configure,
    create_property,
    create_record,
    create_model_relationship,
    valid_organization,
    valid_dataset,
    valid_user,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset
    _, user_node_id = valid_user

    client = configure_client(audit_logger)

    # Create some models:
    patient = model_configure("patient")
    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{patient['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )

    hometown = model_configure("hometown")
    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{hometown['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )

    doctor = model_configure("doctor")
    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{doctor['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            }
        ],
    )

    hospital = model_configure("hospital")
    client.put(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{hospital['id']}/properties",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "modelTitle": True,
            },
            {
                "name": "rating",
                "displayName": "Rating",
                "description": "",
                "dataType": "Long",
            },
        ],
    )

    resides_in = create_model_relationship(  # noqa: F841
        patient, "resides_in", hometown, one_to_many=False
    )
    treats = create_model_relationship(  # noqa: F841
        doctor, "treats", patient, one_to_many=True
    )
    works_at = create_model_relationship(  # noqa: F841
        doctor, "works_at", hospital, one_to_many=False
    )

    springfield = create_record(hometown, {"name": "Springfield"})

    # Make sure record create events are emitted:
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(
            id=UUID(springfield["id"]), name="Springfield", model_id=hometown["id"]
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    shelbyville = create_record(hometown, {"name": "Shelbyville"})
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(
            id=UUID(shelbyville["id"]), name="Shelbyville", model_id=hometown["id"]
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    p1 = create_record(patient, {"name": "Homer"})
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(id=UUID(p1["id"]), name="Homer", model_id=patient["id"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    p2 = create_record(patient, {"name": "Lionel Hutz"})
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(
            id=UUID(p2["id"]), name="Lionel Hutz", model_id=patient["id"]
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    d1 = create_record(doctor, {"name": "Dr. Nick"})
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(id=UUID(d1["id"]), name="Dr. Nick", model_id=doctor["id"]),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    d2 = create_record(doctor, {"name": "Dr. Hibbert"})  # noqa: F841
    jobs_client.send_changelog_event.assert_called_with(
        dataset_id=dataset_id.id,
        event=CreateRecord(
            id=UUID(d2["id"]), name="Dr. Hibbert", model_id=doctor["id"]
        ),
        organization_id=organization_id.id,
        trace_id=trace_id,
        user_id=user_node_id,
    )

    # Create a new record
    h1 = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{hospital['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {"name": "Upstairs Hollywood Medical College", "rating": 10}},
    ).json
    h2 = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{hospital['id']}/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"values": {"name": "Springfield General", "rating": 9}},
    ).json

    p1_resides_in = create_relationship(
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        p1,
        type="resides_in",
        to=springfield["id"],
    )

    assert p1_resides_in["displayName"] == resides_in["displayName"]

    p2_resides_in = create_relationship(
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        p2,
        type="resides_in",
        to=shelbyville["id"],
    )
    assert p2_resides_in["displayName"] == resides_in["displayName"]

    d1_works_at_h1 = create_relationship(
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        d1,
        type="works_at",
        to=h1["id"],
    )
    assert d1_works_at_h1["displayName"] == works_at["displayName"]

    d2_works_at_h2 = create_relationship(
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        d1,
        type="works_at",
        to=h1["id"],
    )
    assert d2_works_at_h2["displayName"] == works_at["displayName"]

    # Deny creating a record-level relationship that doesn't exist at the model level:
    # based on an invalid relationship type:
    with pytest.raises(AssertionError):
        create_relationship(
            client,
            dict(**auth_headers, **trace_id_headers),
            organization_id.id,
            dataset_id.id,
            d1,
            type="foo",
            to=p1["id"],
        )

    with pytest.raises(AssertionError):
        create_relationship(
            client,
            dict(**auth_headers, **trace_id_headers),
            organization_id.id,
            dataset_id.id,
            p1,
            type="attends",
            to=h1["id"],
        )

    # A doctor can treat multiple patients:
    d1_treats_p1 = create_relationship(
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        d1,
        type="treats",
        to=p1["id"],
    )
    d1_treats_p2 = create_relationship(
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        d1,
        type="treats",
        to=p2["id"],
    )

    # But can't work at multiple hospitals, based on the one_to_many=False constraint:
    with pytest.raises(AssertionError):
        create_relationship(
            client,
            dict(**auth_headers, **trace_id_headers),
            organization_id.id,
            dataset_id.id,
            d1,
            type="works_at",
            to=h2["id"],
        )

    # Delete the first relationship:
    delete_relationship(  # noqa: F841
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        d1,
        d1_treats_p1,
    )

    # Looking up d1_treats_p2 works, but d1_treats_p1 fails:
    r = get_relationship(
        client,
        dict(**auth_headers, **trace_id_headers),
        organization_id.id,
        dataset_id.id,
        d1,
        d1_treats_p2,
    )
    assert r["id"] == d1_treats_p2["id"]
    # verify the record relationship has the same display name as that of
    # model relationship it was created from:
    assert r["displayName"] == treats["displayName"]
    assert "createdAt" in r
    assert "createdBy" in r and r["createdBy"] == user_node_id
    assert "updatedAt" in r
    assert "updatedBy" in r and r["updatedBy"] == user_node_id

    with pytest.raises(AssertionError):
        get_relationship(
            client,
            dict(**auth_headers, **trace_id_headers),
            organization_id.id,
            dataset_id.id,
            d1,
            d1_treats_p1,
        )

    # Batch delete d1_treats_p2, d1_works_at_h1, d1_treats_p1 fails
    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{d1['id']}/relationships/delete",
        headers=dict(**auth_headers, **trace_id_headers),
        json=[d1_treats_p2["id"], d1_works_at_h1["id"], d1_treats_p1["id"]],
    )
    assert r.status_code == 207
    assert r.json == [
        {"status": 200, "id": d1_treats_p2["id"]},
        {"status": 200, "id": d1_works_at_h1["id"]},
        {"status": 404, "id": d1_treats_p1["id"]},
    ]

    # Get embedded records for p1, p2
    # Exclude linked records:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{p1['id']}?linked=false",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    p1 = r.json
    assert "resides_in" not in p1["values"]

    # Include linked records:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{p1['id']}?linked=true",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    p1 = r.json
    assert p1["values"]["resides_in"]["id"] == springfield["id"]

    # Exclude linked records:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{p2['id']}?linked=false",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    p2 = r.json
    assert "resides_in" not in p2["values"]

    # Include linked records:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{p2['id']}?linked=true",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    p2 = r.json
    assert p2["values"]["resides_in"]["id"] == shelbyville["id"]

    # Exclude linked records:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{patient['id']}/records?linked=false",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    results = r.json
    p1, p2 = results["results"]
    assert "resides_in" not in p1["values"]
    assert "resides_in" not in p2["values"]

    # Include linked records:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/{patient['id']}/records?linked=true",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    results = r.json
    p1, p2 = results["results"]
    assert p1["values"]["resides_in"]["title"].lower() == "springfield"
    assert p2["values"]["resides_in"]["title"].lower() == "shelbyville"

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 2


def test_simple_querying(
    client,
    auth_headers,
    trace_id_headers,
    sample_patient_db,
    valid_organization,
    valid_dataset,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/query/patient",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "filters": [
                {
                    "model": "patient",
                    "field": "name",
                    "op": op.STARTS_WITH,
                    "argument": "Al",
                }
            ]
        },
    )

    assert r.status_code == 200
    assert r.json[0] == sample_patient_db["records"]["alice"].to_dict()


def test_proxy_packages(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    model,
    valid_organization,
    valid_dataset,
    create_record,
    api_client,
    partitioned_db,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    client = configure_client(audit_logger)

    record = create_record(model, {})

    api_client.get_packages_response = {
        "package_id": {"content": {"nodeId": "N:package:1234", "id": 1234}}
    }

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}/packages",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == {"limit": 100, "offset": 0, "totalCount": 0, "packages": []}

    r = client.post(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}/packages/N:package:1234",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 201
    proxy = r.json
    assert proxy["packageId"] == 1234
    assert proxy["packageNodeId"] == "N:package:1234"

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}/packages",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == {"limit": 100, "offset": 0, "totalCount": 1, "packages": [proxy]}

    r = client.delete(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}/packages/N:package:1234",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == proxy

    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/records/{record['id']}/packages",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == {"limit": 100, "offset": 0, "totalCount": 0, "packages": []}

    # Make sure the package is still reachable:
    partitioned_db.count_packages() == 1
    partitioned_db.get_package(PackageId(1234)) == PackageNodeId("N:package:1234")

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 3


def test_search(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model_configure,
    create_property,
    create_record,
    api_client,
    sample_patient_db,
    movie_db,
    partitioned_db,
):
    client = configure_client(audit_logger)

    _, movie_db = movie_db
    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    alice = sample_patient_db["records"]["alice"]
    patient = sample_patient_db["models"]["patient"]
    patient_properties = partitioned_db.get_properties(patient)

    partitioned_db.create_package_proxy(
        alice, package_id=1234, package_node_id="N:dataset:1234"
    )

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo")
    ]

    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "name",
                    "value": "Charlie",
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json == {
        "models": [],
        "records": [],
        "limit": 25,
        "offset": 0,
        "totalCount": 0,
    }

    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "name",
                    "value": "Alice",
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json == {
        "records": [{"modelId": patient.id, **alice.to_dict()}],
        "models": [
            {
                "id": patient.id,
                "properties": [p.to_dict() for p in patient_properties],
                "dataset": {"id": dataset_id.id, "node_id": dataset_node_id},
            }
        ],
        "limit": 25,
        "offset": 0,
        "totalCount": 1,
    }

    # Dataset not found / allowed
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "datasets": [100],
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "name",
                    "value": "Alice",
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 400

    # Deserialize Boolean value filters correctly
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "person",
            "filters": [
                {
                    "model": "wiki_page",
                    "property": "published",
                    "value": False,
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json["records"] == [
        {"modelId": movie_db["person"].id, **movie_db["Hugo"].to_dict()}
    ]

    # Search arrays with CONTAINS:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "movie",
            "filters": [
                {
                    "model": "movie",
                    "property": "tags",
                    "value": "scifi",
                    "operator": "CONTAINS",
                }
            ],
        },
    )

    assert r.status_code == 200
    assert r.json["totalCount"] == 3

    # Search strings with CONTAINS:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "movie",
            "filters": [
                {
                    "model": "movie",
                    "property": "title",
                    "value": "matrix",
                    "operator": "CONTAINS",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json["totalCount"] == 3

    # Check that ordering works as expected:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records?orderBy=tag_line",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "movie",
            "filters": [
                {
                    "model": "movie",
                    "property": "title",
                    "value": "matrix",
                    "operator": "CONTAINS",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json["totalCount"] == 3
    assert [
        (r["values"]["title"], r["values"]["tag_line"]) for r in r.json["records"]
    ] == [
        ("The Matrix Revolutions", "Everything that has a beginning has an end"),
        ("The Matrix Reloaded", "Free your mind"),
        ("The Matrix", "Welcome to the Real World"),
    ]

    # Check for descending order:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records?orderBy=tag_line&orderDirection=desc",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "movie",
            "filters": [
                {
                    "model": "movie",
                    "property": "title",
                    "value": "matrix",
                    "operator": "CONTAINS",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json["totalCount"] == 3
    assert [
        (r["values"]["title"], r["values"]["tag_line"]) for r in r.json["records"]
    ] == [
        ("The Matrix", "Welcome to the Real World"),
        ("The Matrix Reloaded", "Free your mind"),
        ("The Matrix Revolutions", "Everything that has a beginning has an end"),
    ]

    # Ignore nonexistent columns for sorting, falling back to using the default ordering:
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records?orderBy=doesnotexist&orderDirection=desc",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "movie",
            "filters": [
                {
                    "model": "movie",
                    "property": "title",
                    "value": "matrix",
                    "operator": "CONTAINS",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json["totalCount"] == 3

    # Search packages / files

    api_client.get_packages_response = {
        1234: {
            "content": {
                "nodeId": "N:package:1234",
                "id": 1234,
                "datasetNodeId": dataset_node_id,
                "datasetId": dataset_id.id,
            }
        }
    }

    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/packages",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "name",
                    "value": "Alice",
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json == {
        "limit": 25,
        "offset": 0,
        "totalCount": 1,
        "packages": [
            {
                "content": {
                    "nodeId": "N:package:1234",
                    "id": 1234,
                    "datasetNodeId": dataset_node_id,
                    "datasetId": dataset_id.id,
                }
            }
        ],
    }

    # retrieve records as csv
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records/csv",
        headers=dict(**auth_headers, **trace_id_headers),
        data={
            "data": """{
                "datasets": [],
                "model": "patient",
                "filters": [
                    {
                        "model": "patient",
                        "property": "age",
                        "value": 1,
                        "operator": ">"
                    }
                ]
            }"""
        },
    )

    assert r.status_code == 200
    reader = csv.DictReader(io.StringIO(r.data.decode()))
    assert sorted(list(reader), key=lambda row: row["Name"]) == [
        OrderedDict({"Dataset Name": "Foo", "Name": "Alice", "Age": "34"}),
        OrderedDict({"Dataset Name": "Foo", "Name": "Bob", "Age": "20"}),
    ]

    # Exceptions thrown in the csv download are not handled by the test client.
    # In production they cause the download to bomb out and the file download to
    # be marked as "failed""
    with pytest.raises(InvalidDatasetError):
        r = client.post(
            f"/v2/organizations/{organization_id.id}/search/records/csv",
            headers=dict(**auth_headers, **trace_id_headers),
            data={
                "data": """{
                    "datasets": [100],
                    "model": "patient",
                    "filters": [
                        {
                            "model": "patient",
                            "property": "name",
                            "value": "Alice",
                            "operator": "="
                        }
                    ]
                }"""
            },
        )
        # Need to start consuming the result to trigger the error
        assert r.status_code == 200
        csv.DictReader(io.StringIO(r.data.decode()))

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 12  # last call fails


def test_search_dates(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model_configure,
    create_property,
    create_record,
    partitioned_db,
    api_client,
):
    client = configure_client(audit_logger)

    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo")
    ]

    patient = model_configure("patient")
    create_property(patient["id"], "name", model_title=True)
    create_property(patient["id"], "dob", data_type=dt.Date())

    # Create one date with V2 API, using simple date string
    alice = create_record(patient, {"name": "Alice", "dob": "2019-11-15"})

    # Create one date with V2 API, using zoned date string
    bob = create_record(
        patient, {"name": "Bob", "dob": "2019-11-16T05:00:00.000+04:00"}
    )

    # Create one date with V1 API, using zoned date string
    r = client.post(
        f"/v1/datasets/{dataset_id.id}/concepts/{patient['id']}/instances",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "values": [
                {"name": "name", "value": "Charlie"},
                {"name": "dob", "value": "2019-11-17T08:00:00.000+10:00"},
            ]
        },
    )
    assert r.status_code == 201
    charlie = r.json
    assert charlie["values"][1]["value"] == "2019-11-16T22:00:00"

    # Can search for simple date (no time)
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "dob",
                    "value": "2019-11-15",
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 1
    assert r.json["records"][0]["id"] == alice["id"]

    # Can search by zoned datetime
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "dob",
                    "value": "2019-11-16T05:00:00.000+04:00",
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 1
    assert r.json["records"][0]["id"] == bob["id"]

    # Can search by UTC datetime with no offset
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "dob",
                    "value": "2019-11-16T22:00:00",
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 1
    assert r.json["records"][0]["id"] == charlie["id"]

    # Can order by datetime
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "patient",
            "filters": [
                {
                    "model": "patient",
                    "property": "dob",
                    "value": "2019-11-16",
                    "operator": ">=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 2

    # retrieve records as csv, properly formatting dates
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records/csv",
        headers=dict(**auth_headers, **trace_id_headers),
        data={
            "data": """{
                "datasets": [],
                "model": "patient",
                "filters": []
            }"""
        },
    )

    assert r.status_code == 200
    reader = csv.DictReader(io.StringIO(r.data.decode()))
    assert sorted(list(reader), key=lambda row: row["Name"]) == [
        OrderedDict({"Dataset Name": "Foo", "Dob": "2019-11-15", "Name": "Alice"}),
        OrderedDict({"Dataset Name": "Foo", "Dob": "2019-11-16", "Name": "Bob"}),
        OrderedDict({"Dataset Name": "Foo", "Dob": "2019-11-16", "Name": "Charlie"}),
    ]

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 7


def test_search_scientific_units(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model_configure,
    create_property,
    create_record,
    partitioned_db,
    api_client,
):
    client = configure_client(audit_logger)

    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo")
    ]

    prescription = model_configure("prescription")
    create_property(prescription["id"], "name", model_title=True)
    create_property(prescription["id"], "dose", data_type=dt.Long(unit="mg"))
    create_property(prescription["id"], "count", data_type=dt.Long())

    ibuprofin = create_record(
        prescription, {"name": "Ibuprofin", "dose": 20, "count": 100}
    )
    tylenol = create_record(prescription, {"name": "Tylenol", "dose": 10, "count": 80})

    # Can search the raw numeric value without passing a unit
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "prescription",
            "filters": [
                {
                    "model": "prescription",
                    "property": "dose",
                    "value": 20,
                    "operator": "=",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 1
    assert r.json["records"][0]["id"] == ibuprofin["id"]

    # Can also explicitly filter by unit
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "prescription",
            "filters": [
                {
                    "model": "prescription",
                    "property": "dose",
                    "value": 20,
                    "operator": "=",
                    "unit": "mg",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 1
    assert r.json["records"][0]["id"] == ibuprofin["id"]

    # Unit is strict - if in a different dimension, return nothing
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "prescription",
            "filters": [
                {
                    "model": "prescription",
                    "property": "dose",
                    "value": 20,
                    "operator": "=",
                    "unit": "s",  # milligrams and seconds are not compatible
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 0

    # Unit is a strict filter - if the field does match the unit, return nothing.
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={
            "model": "prescription",
            "filters": [
                {
                    "model": "prescription",
                    "property": "count",
                    "value": 100,
                    "operator": "=",
                    "unit": "mg",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert len(r.json["records"]) == 0

    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models/prescription",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    assert sorted(r.json, key=lambda p: p["name"]) == [
        {
            "dataType": {"type": "Long", "unit": None},
            "displayName": "Count",
            "name": "count",
            "operators": [
                op.EQUALS.value,
                op.NOT_EQUALS.value,
                op.LESS_THAN.value,
                op.LESS_THAN_EQUALS.value,
                op.GREATER_THAN.value,
                op.GREATER_THAN_EQUALS.value,
            ],
        },
        {
            "dataType": {"type": "Long", "unit": "mg"},
            "displayName": "Dose",
            "name": "dose",
            "operators": [
                op.EQUALS.value,
                op.NOT_EQUALS.value,
                op.LESS_THAN.value,
                op.LESS_THAN_EQUALS.value,
                op.GREATER_THAN.value,
                op.GREATER_THAN_EQUALS.value,
            ],
        },
        {
            "dataType": {"type": "String", "format": None},
            "displayName": "Name",
            "name": "name",
            "operators": [op.EQUALS.value, op.NOT_EQUALS.value, op.STARTS_WITH.value],
        },
    ]

    # If no unit is specified, autocomplete all values
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/prescription/dose/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json[0]["values"] == [10, 20]

    # Can autocomplete a specific unit
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/prescription/dose/values?unit=mg",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json[0]["values"] == [10, 20]

    # If a unit is specified, be strict and return nothing
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/prescription/dose/values?unit=s",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 400

    # handle scientific units when downloading csv
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records/csv",
        headers=dict(**auth_headers, **trace_id_headers),
        data={
            "data": """{
                "datasets": [],
                "model": "prescription",
                "filters": []
            }"""
        },
    )

    assert r.status_code == 200
    reader = csv.DictReader(io.StringIO(r.data.decode()))

    assert sorted(list(reader), key=lambda row: row["Name"]) == [
        OrderedDict(
            {
                "Dataset Name": "Foo",
                "Count": "100",
                "Dose (mg)": "20",
                "Name": "Ibuprofin",
            }
        ),
        OrderedDict(
            {"Dataset Name": "Foo", "Count": "80", "Dose (mg)": "10", "Name": "Tylenol"}
        ),
    ]

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 9


def test_search_model_with_id_property(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model_configure,
    create_property,
    create_record,
    partitioned_db,
    api_client,
):
    client = configure_client(audit_logger)

    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo")
    ]

    model_with_id = model_configure("prescription")
    create_property(model_with_id["id"], "name", model_title=True)
    create_property(model_with_id["id"], "id")

    r = create_record(model_with_id, {"name": "Foo", "id": "bar"})

    # Can search the raw numeric value without passing a unit
    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"model": "prescription", "filters": []},
    )

    assert r.status_code == 200
    assert r.json["records"][0]["id"] != "bar"
    assert r.json["records"][0]["values"]["id"] == "bar"

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 1


def test_search_model_with_datetime_array_property(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    model_configure,
    create_property,
    create_record,
    api_client,
):
    client = configure_client(audit_logger)

    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo")
    ]

    model_with_datetime_array = model_configure("prescription")
    create_property(model_with_datetime_array["id"], "name", model_title=True)
    create_property(
        model_with_datetime_array["id"], "refills", data_type=dt.Array(items=dt.Date())
    )

    r = create_record(
        model_with_datetime_array,
        {
            "name": "Foo",
            "refills": ["1986-05-16T00:00:00+00:00", "2003-11-05T00:00:00+00:00"],
        },
    )

    r = client.post(
        f"/v2/organizations/{organization_id.id}/search/records",
        headers=dict(**auth_headers, **trace_id_headers),
        json={"model": "prescription", "filters": []},
    )

    assert r.status_code == 200
    assert r.json["records"][0]["values"]["refills"] == [
        "1986-05-16T00:00:00+00:00",
        "2003-11-05T00:00:00+00:00",
    ]

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 1


def test_autocomplete(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    other_valid_dataset,
    model_configure,
    create_property,
    create_record,
    api_client,
    movie_db,
):
    client = configure_client(audit_logger)

    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    other_dataset_id, other_dataset_node_id = other_valid_dataset

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo"),
        api.Dataset(other_dataset_node_id, other_dataset_id.id, "Bar"),
    ]

    # Add another movie model with overlapping properties:
    movie2 = model_configure("movie", dataset_id=other_dataset_id.id)
    create_property(
        movie2["id"], "title", model_title=True, dataset_id=other_dataset_id.id
    )
    create_property(
        movie2["id"], "rating", data_type=dt.Double(), dataset_id=other_dataset_id.id
    )
    create_property(movie2["id"], "comments", dataset_id=other_dataset_id.id)

    # Create some records for other_dataset `movie`:
    create_record(
        movie2,
        {"title": "Double Indemnity", "rating": 3.5},
        dataset_id=other_dataset_id.id,
    )
    create_record(
        movie2,
        {"title": "Jurassic Park", "rating": 4.6},
        dataset_id=other_dataset_id.id,
    )
    create_record(
        movie2, {"title": "Safe", "rating": 4.75}, dataset_id=other_dataset_id.id
    )

    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200

    # List models for datasets

    assert r.json == {
        "models": [
            {"name": "award", "displayName": "Award"},
            {"name": "genre", "displayName": "Genre"},
            {"name": "imdb_page", "displayName": "IMDB page"},
            {"name": "movie", "displayName": "Movie"},
            {"name": "person", "displayName": "Person"},
            {"name": "wiki_page", "displayName": "Wikipedia page"},
        ]
    }

    # Filter by dataset ID works for an existing dataset:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models?datasetId={dataset_id.id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == {
        "models": [
            {"name": "award", "displayName": "Award"},
            {"name": "genre", "displayName": "Genre"},
            {"name": "imdb_page", "displayName": "IMDB page"},
            {"name": "movie", "displayName": "Movie"},
            {"name": "person", "displayName": "Person"},
            {"name": "wiki_page", "displayName": "Wikipedia page"},
        ]
    }

    # Filtering by dataset ID is a bad request for a non-existing dataset:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models?datasetId=9999",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 400

    # Filter by models related to a root model
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models?relatedTo=wiki_page",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == {
        "models": [
            {"name": "person", "displayName": "Person"},
            {"name": "wiki_page", "displayName": "Wikipedia page"},
        ]
    }

    # Listing properties for a non-existent model results in a 404:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models/doesnotexist",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 404

    # Listing properties for a non-existing dataset is a bad request:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models/movie?datasetId=9999",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 400

    # Listing properties for a model of a dataset (valid_dataset) should work:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models/movie",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    results_all_datasets = r.json
    # De-dupe properties:
    assert (
        len(results_all_datasets) == 9
    )  # datasets: movie (dataset 1, 7 properties) + movie (dataset 2, 3 properties) - 1 duplicate

    movie_db_properties = {}

    for r in results_all_datasets:
        movie_db_properties[r["name"]] = copy(r)
        r["operators"] = set(r["operators"])
    results_all_datasets.sort(key=lambda e: e["name"])

    assert results_all_datasets == [
        {
            "dataType": {"type": "String", "format": None},
            "displayName": "Comments",
            "name": "comments",
            "operators": set(
                [op.EQUALS.value, op.NOT_EQUALS.value, op.STARTS_WITH.value]
            ),
        },
        {
            "dataType": {"type": "Date"},
            "displayName": "Date of release",
            "name": "date_of_release",
            "operators": set(
                [
                    op.EQUALS.value,
                    op.NOT_EQUALS.value,
                    op.LESS_THAN.value,
                    op.LESS_THAN_EQUALS.value,
                    op.GREATER_THAN.value,
                    op.GREATER_THAN_EQUALS.value,
                ]
            ),
        },
        {
            "dataType": {"type": "String", "format": None},
            "displayName": "Description",
            "name": "description",
            "operators": set(
                [op.EQUALS.value, op.NOT_EQUALS.value, op.STARTS_WITH.value]
            ),
        },
        {
            "dataType": {"type": "Double", "unit": None},
            "displayName": "Rating",
            "name": "rating",
            "operators": set(
                [
                    op.EQUALS.value,
                    op.NOT_EQUALS.value,
                    op.LESS_THAN.value,
                    op.LESS_THAN_EQUALS.value,
                    op.GREATER_THAN.value,
                    op.GREATER_THAN_EQUALS.value,
                ]
            ),
        },
        {
            "dataType": {
                "items": {
                    "type": "String",
                    "format": None,
                    "enum": ["Unwatchable", "Poor", "Fair", "Good", "Excellent"],
                },
                "type": "Enum",
            },
            "displayName": "Rating",
            "name": "rating",
            "operators": set([op.CONTAINS.value]),
        },
        {
            "dataType": {"type": "Long", "unit": None},
            "displayName": "Released",
            "name": "released",
            "operators": set(
                [
                    op.EQUALS.value,
                    op.NOT_EQUALS.value,
                    op.LESS_THAN.value,
                    op.LESS_THAN_EQUALS.value,
                    op.GREATER_THAN.value,
                    op.GREATER_THAN_EQUALS.value,
                ]
            ),
        },
        {
            "dataType": {"type": "String", "format": None},
            "displayName": "Tag Line",
            "name": "tag_line",
            "operators": set(
                [op.EQUALS.value, op.NOT_EQUALS.value, op.STARTS_WITH.value]
            ),
        },
        {
            "dataType": {"items": {"format": None, "type": "String"}, "type": "Array"},
            "displayName": "Tags",
            "name": "tags",
            "operators": set([op.CONTAINS.value]),
        },
        {
            "dataType": {"type": "String", "format": None},
            "displayName": "Title",
            "name": "title",
            "operators": set(
                [op.EQUALS.value, op.NOT_EQUALS.value, op.STARTS_WITH.value]
            ),
        },
    ]

    # Get properties in a specific dataset:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models/movie?datasetId={other_dataset_id.id}",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    results_dataset_2 = r.json
    for r in results_dataset_2:
        r["operators"] = set(r["operators"])
    results_dataset_2.sort(key=lambda e: e["name"])

    assert results_dataset_2 == [
        {
            "dataType": {"type": "String", "format": None},
            "displayName": "Comments",
            "name": "comments",
            "operators": set(
                [op.EQUALS.value, op.NOT_EQUALS.value, op.STARTS_WITH.value]
            ),
        },
        {
            "dataType": {"type": "Double", "unit": None},
            "displayName": "Rating",
            "name": "rating",
            "operators": set(
                [
                    op.EQUALS.value,
                    op.NOT_EQUALS.value,
                    op.LESS_THAN.value,
                    op.LESS_THAN_EQUALS.value,
                    op.GREATER_THAN.value,
                    op.GREATER_THAN_EQUALS.value,
                ]
            ),
        },
        {
            "dataType": {"type": "String", "format": None},
            "displayName": "Title",
            "name": "title",
            "operators": set(
                [op.EQUALS.value, op.NOT_EQUALS.value, op.STARTS_WITH.value]
            ),
        },
    ]

    # Check autocomplete values:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/datasets/{dataset_id.id}/models/imdb_page/properties",
        headers=dict(**auth_headers, **trace_id_headers),
    )

    # Booleans
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/imdb_page/published/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json
    suggested[0]["property"]["name"] == "published"
    assert set(suggested[0]["values"]) == set([False, True])

    # Dates
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/date_of_release/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json
    suggested[0]["property"]["name"] == "date_of_release"
    assert set(suggested[0]["values"]) == set(
        ["1986-05-16T00:00:00+00:00", "2003-11-05T00:00:00+00:00"]
    )

    # String suggestion (with prefix)
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/tag_line/values?prefix=ev",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json
    suggested[0]["property"]["name"] == "tag_line"
    assert set(suggested[0]["values"]) == set(
        ["Everything that has a beginning has an end", "Evil has its winning ways"]
    )

    # String suggestion (without prefix)
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/tag_line/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json
    assert len(suggested[0]["values"]) == 10

    # Numbers:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/released/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json
    suggested[0]["property"]["name"] == "released"
    assert set(suggested[0]["values"]) == set([1986, 2003])

    # Array, no prefix
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/tags/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json
    suggested[0]["property"]["name"] == "tags"
    assert set(suggested[0]["values"]) == set(
        [
            "scifi",
            "dystopia",
            "future",
            "dog",
            "noir",
            "new york",
            "lawyer",
            "email",
            "bookstore",
            "murder",
        ]
    )

    # Array, with prefix:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/tags/values?prefix=s",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json
    suggested[0]["property"]["name"] == "tags"
    # Only distinct values should be returned:
    assert list(sorted(suggested[0]["values"])) == [
        "scifi",
        "seattle",
        "spoon",
        "sports",
        "summer",
    ]

    # Enumerations
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/rating/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    suggested = r.json

    suggested[0]["property"]["name"] == "rating"
    assert set(suggested[0]["values"]) == set([3.5, 4.75])

    suggested[1]["property"]["name"] == "rating"
    assert set(suggested[1]["values"]) == set(
        ["Unwatchable", "Poor", "Fair", "Good", "Excellent"]
    )

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 13


def test_autocomplete_only_suggest_models_with_records(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    create_property,
    create_record,
    create_model_relationship,
    model_configure,
    api_client,
    movie_db,
):
    client = configure_client(audit_logger)

    _, movie_db = movie_db
    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo")
    ]

    # A model with no record instances. This should not be suggested
    other_movie = model_configure("other_movie")
    create_property(other_movie["id"], "title", model_title=True)

    similar_to = create_model_relationship(
        movie_db["movie"], "similar_to", other_movie, one_to_many=True
    )

    # List all models for datasets
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.json == {
        "models": [
            {"name": "award", "displayName": "Award"},
            {"name": "genre", "displayName": "Genre"},
            {"name": "imdb_page", "displayName": "IMDB page"},
            {"name": "movie", "displayName": "Movie"},
            {"name": "person", "displayName": "Person"},
            {"name": "wiki_page", "displayName": "Wikipedia page"},
        ]
    }

    # List models related to "movie"
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models?relatedTo=movie",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == {
        "models": [
            {"name": "award", "displayName": "Award"},
            {"name": "genre", "displayName": "Genre"},
            {"name": "movie", "displayName": "Movie"},
            {"name": "person", "displayName": "Person"},
        ]
    }

    # Should not suggest any properties for "other_movie" since the model has no records
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models/other_movie",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    assert r.json == []

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 3


def test_autocomplete_property_deduplication(
    configure_client,
    audit_logger,
    auth_headers,
    trace_id_headers,
    valid_organization,
    valid_dataset,
    other_valid_dataset,
    model_configure,
    create_property,
    create_record,
    api_client,
    movie_db,
):
    client = configure_client(audit_logger)

    organization_id, _ = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    other_dataset_id, other_dataset_node_id = other_valid_dataset

    api_client.get_datasets_response = [
        api.Dataset(dataset_node_id, dataset_id.id, "Foo"),
        api.Dataset(other_dataset_node_id, other_dataset_id.id, "Bar"),
    ]

    # Add another movie model with overlapping properties:
    movie2 = model_configure("movie", dataset_id=other_dataset_id.id)

    # Has the same name and type as the other movie "title" property:
    create_property(
        movie2["id"], "title", model_title=True, dataset_id=other_dataset_id.id
    )
    # Has the same name but a diffent type as the other movie "rating" property:
    create_property(
        movie2["id"], "rating", data_type=dt.Double(), dataset_id=other_dataset_id.id
    )

    # Create some records for other_dataset `movie`:
    create_record(
        movie2,
        {"title": "Double Indemnity", "rating": 3.5},
        dataset_id=other_dataset_id.id,
    )
    create_record(
        movie2,
        {"title": "Jurassic Park", "rating": 4.6},
        dataset_id=other_dataset_id.id,
    )

    # Check properties are de-duped:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/models/movie",
        headers=dict(**auth_headers, **trace_id_headers),
    )

    assert r.status_code == 200
    property_names = list(sorted(s["name"] for s in r.json))
    # rating shoudl appear twice, as the datatypes differ, but title should appear
    # once, as the other title/String instace should be removed.
    assert property_names == [
        "date_of_release",
        "description",
        "rating",
        "rating",
        "released",
        "tag_line",
        "tags",
        "title",
    ]

    # Check property value entries are de-deduped:
    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/title/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    assert r.status_code == 200
    results = r.json
    assert len(results) == 1 and results[0]["property"]["name"] == "title"
    assert set(results[0]["values"]) == {
        "You've Got Mail",
        "Snow Falling on Cedars",
        "What Dreams May Come",
        "As Good as It Gets",
        "Stand By Me",
        "Jerry Maguire",
        "Top Gun",
        "A Few Good Men",
        "The Devil's Advocate",
        "Sleepless in Seattle",
        "Jurassic Park",
        "Double Indemnity",
    }

    r = client.get(
        f"/v2/organizations/{organization_id.id}/autocomplete/movie/rating/values",
        headers=dict(**auth_headers, **trace_id_headers),
    )
    results = r.json
    assert len(results) == 2
    results.sort(
        key=lambda r: str(r["property"]["dataType"])
    )  # for deterministic order
    assert results[0]["values"] == ["Unwatchable", "Poor", "Fair", "Good", "Excellent"]
    assert results[1]["values"] == [3.5, 4.6]

    assert audit_logger.enhance.called
    assert audit_logger.enhance.call_count == 3
