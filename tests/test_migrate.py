from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import pytest
import requests
from blackfynn import Blackfynn
from blackfynn.models import DataPackage, ModelProperty
from requests.exceptions import HTTPError

from migrate.core import (
    PennsieveDatabase,
    delete_orphaned_datasets_impl,
    migrate_dataset,
)


@dataclass
class MockDataset:
    id: int
    node_id: str
    name: str
    state: str


class MockPennsieveDatabase(PennsieveDatabase):
    def __init__(
        self,
    ):
        self.get_organizations_response = []
        self.get_dataset_response = None
        self.get_dataset_ids_response = []
        self._locked_datasets = set()

    def get_organizations(self):
        return self.get_organizations_response

    def get_dataset(self, organization_id: int, dataset_id: int):
        return self.get_dataset_response

    def get_dataset_ids(self, organization_id: int):
        return self.get_dataset_ids_response

    def lock_dataset(self, organization_id: int, dataset_id: int):
        self._locked_datasets.add(dataset_id)

    def unlock_dataset(self, organization_id: int, dataset_id: int):
        self._locked_datasets.add(dataset_id)


@pytest.fixture
def bf():
    return Pennsieve()


@pytest.fixture(scope="function")
def dataset(bf):
    """
    Test Dataset to be used by other tests.
    """
    ds = bf.create_dataset("test_dataset_{}".format(uuid4()))

    yield ds

    bf._api.datasets.delete(ds)


@pytest.fixture
def organization(bf):
    organization = bf._api.organizations.get(bf._api._organization)
    try:
        data = bf._api._get(f"/models/v2/organizations/{organization.int_id}")
        assert data["service"] == "Neptune"
    except HTTPError as e:
        assert e.response.status_code == 404, "Organization must still be in Neptune"

    return organization


def mock_pennsieve_dataset():
    return MockPennsieveDatabase()


@pytest.fixture(scope="function")
def assert_in_neo4j(bf, organization, dataset):
    # Dataset should now be in Neo4j

    def check():
        # Drop down to raw `requests` to access the response headers
        r = requests.get(
            f"{bf._api.settings.api_host}/models/v2/organizations/{organization.int_id}/datasets/{dataset.int_id}/models",
            headers=bf._api.session.headers,
        )

        assert r.status_code == 200
        assert r.headers["x-model-service"] == "Neo4j"

    return check


@pytest.mark.skip
@pytest.mark.integration
def test_migrate_dataset(bf, dataset, organization, assert_in_neo4j):
    """
    Needs to be run in non-prod, on an organization that has not been fully migrated.
    """
    person = dataset.create_model(
        "Person",
        schema=[
            ModelProperty("name", data_type=str, title=True, required=True),
            ModelProperty("age", data_type=int),
            ModelProperty("dob", data_type=datetime),
        ],
    )

    likes = dataset.create_relationship_type(
        "likes", "", source=person.id, destination=person.id
    )

    alice = person.create_record(
        {"name": "Alice", "age": 25, "dob": datetime(1994, 11, 19)}
    )

    bob = person.create_record(
        {"name": "Bob", "age": 24, "dob": datetime(1995, 11, 19)}
    )

    likes.relate(alice, bob)

    migrate_dataset(
        organization_id=organization.int_id,
        #        organization_node_id=organization.id,
        dataset_ids=[dataset.int_id]
        #        dataset_node_id=dataset.id,
    )

    assert_in_neo4j()

    assert len(dataset.models()) == 1
    person = dataset.get_model("Person")
    assert person.get_all() == [alice, bob]

    assert len(dataset.relationships()) == 1
    assert alice.get_related(person.type) == [bob]


@pytest.mark.skip
@pytest.mark.integration
def test_merge_duplicate_linked_property_instances(
    bf, dataset, organization, assert_in_neo4j
):
    """
    Needs to be run in non-prod, on an organization that has not been fully migrated.

    Test that the migration dedupulicates linked property instances.
    """
    person = dataset.create_model(
        "Person",
        schema=[ModelProperty("name", data_type=str, title=True, required=True)],
    )
    likes = person.add_linked_property("likes", person)

    alice = person.create_record({"name": "Alice"})

    bob = person.create_record({"name": "Bob"})

    # Add one instance: (alice)-[likes]->(alice)
    alice.add_linked_value(alice, likes)

    # Add another: (alice)-[likes]->(bob)
    # The Python client deletes existing linked properties before adding new ones -
    # this manual request allows us to duplicate the linked property instance.
    # (This is the same requeust `add_linked_value` makes under the hood.)
    bf._api.concepts.instances._post(
        bf._api.concepts.instances._uri(
            "/{dataset_id}/concepts/{concept_id}/instances/{instance_id}/linked",
            dataset_id=dataset.id,
            concept_id=person.id,
            instance_id=alice.id,
        ),
        json={
            "schemaLinkedPropertyId": likes.id,
            "to": bob.id,
            "name": "likes",
            "displayName": "likes",
        },
    )

    # This should not be possible - there should only be one "likes" property
    assert [lp.type.name for lp in alice.get_linked_values()] == ["likes", "likes"]

    migrate_dataset(organization_id=organization.int_id, dataset_ids=[dataset.int_id])

    assert_in_neo4j()

    # Import should merge linked values
    linked_values = alice.get_linked_values()
    assert len(linked_values) == 1
    assert linked_values[0].type.name == "likes"
    assert linked_values[0].target_record_id == bob.id


@pytest.mark.skip
@pytest.mark.integration
def test_merge_model_relationships(bf, dataset, organization, assert_in_neo4j):
    """
    Needs to be run in non-prod, on an organization that has not been fully migrated.

    Test that model and schema relationships are merged in Neo4j.
    """
    person = dataset.create_model(
        "Person",
        schema=[ModelProperty("name", data_type=str, title=True, required=True)],
    )

    food = dataset.create_model(
        "Food", schema=[ModelProperty("name", data_type=str, title=True, required=True)]
    )

    color = dataset.create_model(
        "Color",
        schema=[ModelProperty("name", data_type=str, title=True, required=True)],
    )

    # Relationship type with no "from" and "to"
    likes = dataset.create_relationship_type("Likes", "likes")

    # Relationship type with "from" and "to", but no instances
    dataset.create_relationship_type(
        "Appreciates", "appreciates", source=person.id, destination=color.id
    )

    alice = person.create_record({"name": "Alice"})
    bob = person.create_record({"name": "Bob"})
    charlie = person.create_record({"name": "Charlie"})

    ice_cream = food.create_record({"name": "Ice Cream"})

    alice_likes_bob = likes.relate(alice, bob)
    bob_likes_charlie = likes.relate(bob, charlie)
    alice_likes_ice_cream = likes.relate(alice, ice_cream)

    # At this point we have in the relation_types file
    #
    #  ()-[likes]->()
    #  (person)-[appreciates]->(color)
    #
    # and in the schemaRelations file
    #
    #  (person)-[likes]->(person)
    #  (person)-[likes]->(food)
    #
    # The /relationships endpoint on the old service *only* returns things in
    # the relation_types file.
    #
    # But the new service should merge them both together to create all
    # necessary model relationships and stubs:
    #
    #  ()-[likes]->()
    #  (person)-[appreciates]->(color)
    #  (person)-[likes]->(person)
    #  (person)-[likes]->(food)

    migrate_dataset(
        organization_id=organization.int_id,
        #        organization_node_id=organization.id,
        dataset_ids=[dataset.int_id]
        #        dataset_node_id=dataset.id,
    )

    assert_in_neo4j()

    # Drop into raw requests because of
    # https://app.clickup.com/t/426zh9
    relationships = bf._api.concepts.relationships._get(
        bf._api.concepts.relationships._uri(
            "/{dataset_id}/relationships", dataset_id=dataset.id
        )
    )

    assert sorted(
        [(r["from"] or "*", r["name"], r["to"] or "*") for r in relationships]
    ) == sorted(
        [
            ("*", "Likes", "*"),
            (person.id, "Likes", food.id),
            (person.id, "Likes", person.id),
            (person.id, "Appreciates", color.id),
        ]
    )


@pytest.mark.skip
@pytest.mark.integration
def test_package_proxy_import(bf, dataset, organization, assert_in_neo4j):
    """
    Needs to be run in non-prod, on an organization that has not been fully migrated.

    Test that multiple proxy relationships can be imported for the same proxy
    concept, and that we can import edges both too and from proxy concepts.
    """
    person = dataset.create_model(
        "Person",
        schema=[ModelProperty("name", data_type=str, title=True, required=True)],
    )

    alice = person.create_record({"name": "Alice"})
    bob = person.create_record({"name": "Bob"})

    pkg = DataPackage("Some MRI", package_type="MRI")
    dataset.add(pkg)

    pkg.relate_to(alice, bob)

    alice_files = alice.get_files()
    bob_files = bob.get_files()

    migrate_dataset(organization_id=organization.int_id, dataset_ids=[dataset.int_id])

    assert_in_neo4j()

    assert alice.get_files() == alice_files
    assert bob.get_files() == bob_files


@pytest.mark.skip
@pytest.mark.integration
def test_migrate_default_values(bf, dataset, organization, assert_in_neo4j):
    """
    Needs to be run in non-prod, on an organization that has not been fully migrated.
    """
    person = dataset.create_model("Person")

    bf._api.concepts._put(
        bf._api.concepts._uri(
            "/{dataset_id}/concepts/{concept_id}/properties",
            dataset_id=dataset.id,
            concept_id=person.id,
        ),
        json=[
            {
                "name": "name",
                "displayName": "Name",
                "description": "",
                "dataType": "String",
                "locked": False,
                "conceptTitle": True,
                "required": False,
                "default": True,
            },
            {
                "name": "favorite_colors",
                "displayName": "Favorite Colors",
                "description": "",
                "dataType": {"type": "Array", "items": {"type": "String"}},
                "locked": False,
                "conceptTitle": False,
                "required": False,
                "default": True,
                "defaultValue": ["red", "green", "blue"],
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
                "defaultValue": 39,
            },
        ],
    )

    migrate_dataset(organization_id=organization.int_id, dataset_ids=[dataset.int_id])

    assert_in_neo4j()

    r = bf._api.concepts._get(
        bf._api.concepts._uri(
            "/{dataset_id}/concepts/{concept_id}/properties",
            dataset_id=dataset.id,
            concept_id=person.id,
        )
    )

    assert [
        (
            p["name"],
            set(p["defaultValue"])
            if isinstance(p["defaultValue"], list)
            else p["defaultValue"],
        )
        for p in r
    ] == [("name", None), ("favorite_colors", {"red", "green", "blue"}), ("age", 39)]


def test_delete_skip_datasets_still_in_api(large_partitioned_db):
    organization_id = large_partitioned_db.organization_id
    bf_database = mock_pennsieve_dataset()

    before_summary = large_partitioned_db.summarize()
    assert before_summary.model_count > 0
    assert before_summary.model_record_count > 0

    bf_database.get_dataset_ids_response = [large_partitioned_db.dataset_id]
    db = large_partitioned_db._unrestricted
    delete_orphaned_datasets_impl(bf_database, db, organization_id, dry_run=False)

    after_summary = large_partitioned_db.summarize()
    assert before_summary.model_count == after_summary.model_count
    assert before_summary.model_record_count == after_summary.model_record_count


def test_delete_dataset_fails_if_not_deleted(config, large_partitioned_db):
    organization_id = large_partitioned_db.organization_id
    bf_database = mock_pennsieve_dataset()

    bf_database.get_dataset_ids_response = []
    bf_database.get_dataset_response = MockDataset(
        large_partitioned_db.dataset_id, "N:some-id", "Fake", "READY"
    )
    db = large_partitioned_db._unrestricted

    with pytest.raises(AssertionError):
        delete_orphaned_datasets_impl(bf_database, db, organization_id, dry_run=False)


def test_delete_orphaned_datasets(config, large_partitioned_db):
    organization_id = large_partitioned_db.organization_id
    bf_database = mock_pennsieve_dataset()

    before_summary = large_partitioned_db.summarize()
    assert before_summary.model_count > 0
    assert before_summary.model_record_count > 0

    bf_database.get_dataset_ids_response = []
    db = large_partitioned_db._unrestricted
    delete_orphaned_datasets_impl(bf_database, db, organization_id, dry_run=False)

    after_summary = large_partitioned_db.summarize()
    assert after_summary.model_count == 0
    assert after_summary.model_record_count == 0
