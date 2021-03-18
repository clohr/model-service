import dataclasses
import logging
from datetime import datetime, timezone
from time import sleep
from typing import Dict, Iterator, List, Tuple, Union, cast
from uuid import UUID, uuid4

import pytest
from more_itertools import unique_everseen

from core.dtos.api import Dataset as DatasetDTO
from core.util import normalize_datetime
from server import models as m
from server.db import (
    EmbedLinked,
    PartitionedDatabase,
    QueryRunner,
    SearchDatabase,
    labels,
)
from server.db.util import match_clause
from server.errors import (
    ExceededTimeLimitError,
    ModelNotFoundError,
    ModelRelationshipNotFoundError,
)
from server.models import (
    Dataset,
    GraphValue,
    Model,
    ModelId,
    ModelProperty,
    ModelRelationship,
    Package,
    Record,
    RecordStub,
)
from server.models import datatypes as dt
from server.models.query import GroupCount
from server.models.query import Operator as op
from server.models.query import OrderBy, UserQuery
from server.models.search import (
    DatasetFilter,
    ModelFilter,
    ModelSuggestion,
    PackageSearchResult,
    PropertyFilter,
    SearchResult,
)

log = logging.getLogger(__file__)


def create_test_datasetDTO(partitioned_db: PartitionedDatabase, name: str = "Foo"):
    return DatasetDTO(partitioned_db.dataset_node_id, partitioned_db.dataset_id, name)


def get_organization_and_dataset(db):
    return db.execute_single(
        f"""
        MATCH ({labels.organization("o")} {{ id: $organization_id }})
        OPTIONAL MATCH ({labels.dataset("d")} {{ id: $dataset_id }})-[{labels.in_organization()}]->(o)
        RETURN o AS organization, d AS dataset
        """,
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
    ).single()


def get_orphaned_nodes(db):
    return list(
        db.execute_single(
            f"""
            MATCH (x)
            WHERE NOT '{labels.ORGANIZATION_LABEL}' IN LABELS(x)
            AND NOT EXISTS((x)-[*6]-({labels.organization()}))
            RETURN x
            """
        ).records()
    )


def test_node_ids_set_upon_dataset_and_organization_node_creation(
    partitioned_db, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    result = get_organization_and_dataset(partitioned_db)
    assert result is None  # No nodes created until a model is created

    _ = partitioned_db.create_model("bicycle", "Bicycle", "A bicycle")

    result = get_organization_and_dataset(partitioned_db)
    assert result is not None
    assert result["organization"]["id"] == organization_id.id
    assert result["organization"]["node_id"] == organization_node_id
    assert result["dataset"]["id"] == dataset_id.id
    assert result["dataset"]["node_id"] == dataset_node_id


def test_multiple_organizations_can_create_dataset_with_same_integer_id(
    partitioned_db,
    valid_organization,
    other_valid_organization,
    valid_dataset,
    valid_user,
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset
    _, user_node_id = valid_user

    other_organization_id, other_organization_node_id = other_valid_organization

    _ = partitioned_db.create_model("bicycle", "Bicycle", "A bicycle")

    other_organization_db = PartitionedDatabase(
        db=partitioned_db.db,
        organization_id=other_organization_id.id,
        dataset_id=dataset_id.id,
        user_id=user_node_id,
        organization_node_id=other_organization_node_id,
        dataset_node_id="N:dataset:a-different-dataset",
    )

    _ = other_organization_db.create_model("patient", "Patient", "A patient")

    result = get_organization_and_dataset(partitioned_db)
    assert result["organization"]["id"] == organization_id.id
    assert result["organization"]["node_id"] == organization_node_id
    assert result["dataset"]["id"] == dataset_id.id
    assert result["dataset"]["node_id"] == dataset_node_id

    result = get_organization_and_dataset(other_organization_db)
    assert result["organization"]["id"] == other_organization_id.id
    assert result["organization"]["node_id"] == other_organization_node_id
    assert result["dataset"]["id"] == dataset_id.id
    assert result["dataset"]["node_id"] == "N:dataset:a-different-dataset"


def test_create_record_with_user_defined_id_property(partitioned_db):
    bike = partitioned_db.create_model("bicycle", "Bicycle", "A bicycle")
    partitioned_db.update_properties(
        bike,
        ModelProperty(
            name="id",
            display_name="ID",
            data_type=dt.String(),
            description="Identification number",
            model_title=True,
        ),
    )
    record = partitioned_db.create_records(
        bike, [{"id": "c90a3931-f212-46e1-b751-af09b4be9e47"}]
    )[0]
    assert record.id != "c90a3931-f212-46e1-b751-af09b4be9e47"
    assert isinstance(record.id, UUID)
    assert record.values["id"] == "c90a3931-f212-46e1-b751-af09b4be9e47"


def create_test_records(
    partitioned_db: PartitionedDatabase,
    model_id_or_name: Union[Model, ModelId, str],
    records: List[Dict[str, GraphValue]],
) -> List[Record]:
    kwargs: Dict[str, GraphValue] = cast(
        Dict[str, GraphValue],
        dict(
            organization_id=partitioned_db.organization_id,
            dataset_id=partitioned_db.dataset_id,
            user_id=str(partitioned_db.user_id),
            records=records,
        ),
    )

    # See the note on `create_models()` regarding a description of how
    # the record `@sort_key` property works.
    cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("match_id_or_name", model_id_or_name, kwargs)} }})
              -[{labels.in_dataset()}]->({{ id: $dataset_id }})
              -[{labels.in_organization()}]->({{ id: $organization_id }})
        UNWIND $records as record
        CREATE ({labels.record("r")} {{
          `@sort_key`: 0,
          `@id`: randomUUID()
        }})
        SET r += record
        SET m.`@max_sort_key` = m.`@max_sort_key` + 1
        SET r.`@sort_key` = m.`@max_sort_key`
        CREATE (r)-[{labels.instance_of()}]->(m)

        MERGE ({labels.user("u")} {{ node_id: $user_id }})
        CREATE (r)-[{labels.created_by("created")} {{ at: datetime() }}]->(u)
        CREATE (r)-[{labels.updated_by("updated")} {{ at: datetime() }}]->(u)

        RETURN COUNT(r) as count
        """
    result = partitioned_db.execute_single(cql, **kwargs).single()

    return result["count"]


def get_all_test_records(
    partitioned_db: PartitionedDatabase, model_id_or_name: str
) -> List[Record]:
    kwargs: Dict[str, GraphValue] = dict(
        organization_id=partitioned_db.organization_id,
        dataset_id=partitioned_db.dataset_id,
    )

    cql = f"""
        MATCH  ({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        RETURN r
        """

    nodes = partitioned_db.execute_single(cql, **kwargs).records()

    if nodes is None:
        raise ModelNotFoundError(model_id_or_name)

    return nodes


# this is a performance test.  when run with -s, we should see that we can modify 100k records in ~5s
@pytest.mark.skip(
    reason="this test is not run in CI because it is slow and intended to measure performance.  When it is run manually, it should pass."
)
def test_delete_property_from_all_records(partitioned_db):
    bike = partitioned_db.create_model("bicycle", "Bicycle", "A bicycle")
    description = ModelProperty(
        name="description",
        display_name="Description",
        data_type=dt.String(),
        description="Description",
    )
    partitioned_db.update_properties(
        bike,
        ModelProperty(
            name="id",
            display_name="ID",
            data_type=dt.String(),
            description="Identification number",
            model_title=True,
        ),
        ModelProperty(
            name="name", display_name="Name", data_type=dt.String(), description="Name"
        ),
        description,
    )

    count = 100000
    batch = 10000
    iterations = int(count / batch)

    records = []
    for i in range(count):
        records.append({"id": str(uuid4()), "name": str(i), "description": str(i)})

    before = datetime.now()
    saved = 0
    for i in range(iterations):
        start = i * batch
        end = (i + 1) * batch
        saved += create_test_records(partitioned_db, bike, records[start:end])
    after = datetime.now()
    print(
        f"""record creation: {(after.timestamp() - before.timestamp()) * 1000} millis"""
    )

    original = get_all_test_records(partitioned_db, bike.id)

    original_count = 0
    for record in original:
        assert "description" in record["r"]
        original_count += 1

    assert original_count == count

    print(f"""{count} records created, all have description property""")

    before = datetime.now()
    prop_deleted = partitioned_db.delete_property_from_all_records(bike.id, description)
    after = datetime.now()

    print(
        f"""property removal: {(after.timestamp() - before.timestamp()) * 1000} millis"""
    )

    assert prop_deleted == count

    updated = get_all_test_records(partitioned_db, bike.id)

    updated_count = 0
    for record in updated:
        assert "description" not in record["r"]
        updated_count += 1

    assert updated_count == count

    print(f"""{count} records updated, all have description property removed""")


def test_get_all_records_ordered_by_sort_key(partitioned_db):
    bike = partitioned_db.create_model("bicycle", "Bicycle", "A bicycle")
    partitioned_db.update_properties(
        bike,
        ModelProperty(
            name="brand",
            display_name="Brand",
            data_type=dt.String(),
            description="Brand",
            model_title=True,
        ),
        ModelProperty(
            name="year",
            display_name="Year",
            data_type=dt.Long(),
            description="Year of manufacture",
        ),
        ModelProperty(
            name="frame_size",
            display_name="Frame size (cm)",
            data_type=dt.Double(unit="cm"),
            description="Standover frame height",
        ),
    )

    partitioned_db.create_records(
        bike,
        [
            {"brand": "Bianchi", "year": 1999, "frame_size": 54},
            {"brand": "Colnago", "year": 1977, "frame_size": 56},
            {"brand": "Cinelli", "year": 1986, "frame_size": 56},
            {"brand": "Guerciotti", "year": 1981, "frame_size": 57},
            {"brand": "Ciocc", "year": 1978, "frame_size": 56},
            {"brand": "De Rosa", "year": 1992, "frame_size": 53},
        ],
    )

    def get_values(page):
        return [p["values"] for p in page.to_dict()["results"]]

    page1 = partitioned_db.get_all_records(bike, limit=3)
    page2 = partitioned_db.get_all_records(bike, next_page=page1.next_page)

    items = get_values(page1) + get_values(page2)

    # Assert that the items come back in the same order they were created in.
    # The Record objects returned by `get_all_records` does not include the
    # @sort_key directly, but if it is working correctly, the records will
    # come back in the same order:
    assert items[0]["brand"].lower() == "bianchi"
    assert items[1]["brand"].lower() == "colnago"
    assert items[2]["brand"].lower() == "cinelli"
    assert items[3]["brand"].lower() == "guerciotti"
    assert items[4]["brand"].lower() == "ciocc"
    assert items[5]["brand"].lower() == "de rosa"


def test_query(partitioned_db):
    patient = partitioned_db.create_model("patient", "Patient", "a person")
    partitioned_db.update_properties(
        patient,
        ModelProperty(
            name="name",
            display_name="Name",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
        ModelProperty(name="age", display_name="Age", data_type=dt.Long()),
    )

    visit = partitioned_db.create_model("visit", "Visit", "a visit")
    partitioned_db.update_properties(
        visit,
        ModelProperty(
            name="day",
            display_name="Day",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
    )

    medication = partitioned_db.create_model("medication", "Medication", "a medication")
    partitioned_db.update_properties(
        medication,
        ModelProperty(
            name="name",
            display_name="Name",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
    )

    attends = partitioned_db.create_model_relationship(patient, "attends", visit)
    prescribed = partitioned_db.create_model_relationship(
        visit, "prescribed", medication
    )

    alice, bob = partitioned_db.create_records(
        patient, [{"name": "Alice", "age": 34}, {"name": "Bob", "age": 20}]
    )
    monday, tuesday = partitioned_db.create_records(
        visit, [{"day": "Monday"}, {"day": "Tuesday"}]
    )
    aspirin, tylenol = partitioned_db.create_records(
        medication, [{"name": "Aspirin"}, {"name": "Tylenol"}]
    )

    partitioned_db.create_record_relationship(alice, attends, monday)
    partitioned_db.create_record_relationship(monday, prescribed, aspirin)

    partitioned_db.create_record_relationship(bob, attends, tuesday)
    partitioned_db.create_record_relationship(tuesday, prescribed, aspirin)
    partitioned_db.create_record_relationship(tuesday, prescribed, tylenol)

    assert sorted(QueryRunner(partitioned_db).run(visit)) == sorted([monday, tuesday])

    assert sorted(QueryRunner(partitioned_db).run(medication)) == sorted(
        [aspirin, tylenol]
    )

    q = UserQuery().with_filter(patient, field="name", op=op.EQUALS, argument="Alice")
    assert QueryRunner(partitioned_db, q).run(medication) == [aspirin]

    # Numerical properties
    q = UserQuery().with_filter(
        model=patient, field="age", op=op.LESS_THAN, argument=25
    )
    assert sorted(QueryRunner(partitioned_db, q).run(medication)) == sorted(
        [aspirin, tylenol]
    )

    # Negation
    q = UserQuery().with_filter(
        "patient", field="age", op=op.LESS_THAN, argument=25, negate=True
    )
    assert sorted(QueryRunner(partitioned_db, q).run("medication")) == [aspirin]

    # Starts with
    q = UserQuery().with_filter(patient, field="name", op=op.STARTS_WITH, argument="Al")
    assert QueryRunner(partitioned_db, q).run(medication) == [aspirin]

    # Contains
    q = UserQuery().with_filter(patient, field="name", op=op.CONTAINS, argument="ic")
    assert QueryRunner(partitioned_db, q).run(medication) == [aspirin]


def test_query_movie_db(partitioned_db, movie_db):
    q = (
        UserQuery()
        .with_filter("award", field="name", op=op.EQUALS, argument="BAFTA Award")
        .with_filter("genre", field="name", op=op.EQUALS, argument="Action")
        .with_filter("person", field="name", op=op.STARTS_WITH, argument="Keanu")
    )

    results = QueryRunner(partitioned_db, q).run("movie")

    assert len(results) == 1
    assert results[0].values["title"] == "The Matrix"  # Real value
    assert (
        results[0].values["description"] == "--NO DESCRIPTION PROVIDED--"
    )  # Default value


def test_querying_movie_db_sorting(partitioned_db, movie_db):
    q = (
        UserQuery()
        .with_filter("person", field="name", op=op.EQUALS, argument="Tom Cruise")
        .connect_to("movie")
        .order_by(OrderBy(field="title", ascending=True))
    )

    results = QueryRunner(partitioned_db, q).run("movie")

    assert len(results) == 3
    assert results[0].values["title"] == "A Few Good Men"
    assert results[1].values["title"] == "Jerry Maguire"
    assert results[2].values["title"] == "Top Gun"


def test_querying_movie_db_with_linked_records(partitioned_db, movie_db):
    q = (
        UserQuery()
        .with_filter("award", field="name", op=op.EQUALS, argument="Academy Award")
        .with_filter("genre", field="name", op=op.EQUALS, argument="Action")
    )

    # Exclude linked records:
    results = QueryRunner(partitioned_db, q).run("person")
    results = sorted(results, key=lambda r: r.values["name"])

    assert results[0].values["name"] == "Lana Wachowski"
    assert "has_imdb_entry" not in results[0].values
    assert "has_wiki_entry" not in results[0].values
    assert results[1].values["name"] == "Lilly Wachowski"
    assert "has_imdb_entry" not in results[0].values
    assert "has_wiki_entry" not in results[0].values

    # Include linked records:
    results = QueryRunner(partitioned_db, q).run(
        "person", embed_linked=EmbedLinked.STUB
    )
    results = sorted(results, key=lambda r: r.values["name"])

    assert len(results) == 2

    assert results[0].values["name"] == "Lana Wachowski"

    assert isinstance(results[0].values["has_imdb_entry"], RecordStub)
    assert isinstance(results[0].values["has_wiki_entry"], RecordStub)

    assert results[1].values["name"] == "Lilly Wachowski"
    assert isinstance(results[1].values["has_imdb_entry"], RecordStub)
    assert isinstance(results[1].values["has_wiki_entry"], RecordStub)


def test_query_movie_db_aggregates(partitioned_db, movie_db):
    q = (
        UserQuery()
        .with_filter("person", field="born", op=op.GREATER_THAN, argument=1960)
        .aggregate(GroupCount("born", "person"))
    )

    assert QueryRunner(partitioned_db, q).run("person", limit=100) == [
        {
            1962: 6,
            1967: 4,
            1968: 2,
            1973: 1,
            1970: 2,
            1971: 3,
            1963: 2,
            1966: 2,
            1969: 2,
            1972: 1,
            1961: 4,
            1996: 1,
            1974: 1,
            1964: 1,
            1965: 1,
            1978: 1,
            1975: 1,
        }
    ]

    # Limit should not change the GroupCount
    assert QueryRunner(partitioned_db, q).run("person", limit=1) == [
        {
            1962: 6,
            1967: 4,
            1968: 2,
            1973: 1,
            1970: 2,
            1971: 3,
            1963: 2,
            1966: 2,
            1969: 2,
            1972: 1,
            1961: 4,
            1996: 1,
            1974: 1,
            1964: 1,
            1965: 1,
            1978: 1,
            1975: 1,
        }
    ]


def test_user_provenance_for_models(partitioned_db, other_partitioned_db):
    bike = partitioned_db.create_model("bike", "Bicycle", "A bicycle")

    assert bike.created_by == partitioned_db.user_id
    assert bike.updated_by == partitioned_db.user_id
    assert bike.created_at == bike.updated_at

    moto = other_partitioned_db.update_model(
        "bike", "bike", "Motorcyle", "Actually, a motorcycle"
    )

    assert moto.created_by == partitioned_db.user_id
    assert moto.updated_by == other_partitioned_db.user_id
    assert moto.created_at == bike.created_at
    assert moto.updated_at > bike.updated_at

    moto2 = other_partitioned_db.get_model("bike")

    assert moto2.created_by == partitioned_db.user_id
    assert moto2.updated_by == other_partitioned_db.user_id
    assert moto2.created_at == moto.created_at
    assert moto2.updated_at == moto.updated_at


def test_user_provenance_for_properties(partitioned_db, other_partitioned_db):
    bike = partitioned_db.create_model("bike", "Bicycle", "A bicycle")
    brand = partitioned_db.update_properties(
        bike,
        ModelProperty(
            name="brand",
            display_name="Brand",
            data_type=dt.String(),
            description="Brand",
            model_title=True,
        ),
    )[0]

    assert brand.created_by == partitioned_db.user_id
    assert brand.updated_by == partitioned_db.user_id
    assert brand.created_at == brand.updated_at

    make = other_partitioned_db.update_properties(
        bike, dataclasses.replace(brand, display_name="Make", description="Make")
    )[0]

    assert make.created_by == partitioned_db.user_id
    assert make.updated_by == other_partitioned_db.user_id
    assert make.created_at == brand.created_at
    assert make.updated_at > brand.updated_at

    make2 = partitioned_db.get_properties(bike)[0]

    assert make2.created_by == partitioned_db.user_id
    assert make2.updated_by == other_partitioned_db.user_id
    assert make2.created_at == brand.created_at
    assert make2.updated_at == make.updated_at


def test_user_provenance_for_records(partitioned_db, other_partitioned_db):
    bike = partitioned_db.create_model("bike", "Bicycle", "A bicycle")
    partitioned_db.update_properties(
        bike,
        ModelProperty(
            name="brand",
            display_name="Brand",
            data_type=dt.String(),
            description="Brand",
            model_title=True,
        ),
    )
    bianchi = partitioned_db.create_records(bike, [{"brand": "Bianchi"}])[0]

    assert bianchi.created_by == partitioned_db.user_id
    assert bianchi.updated_by == partitioned_db.user_id
    assert bianchi.created_at == bianchi.updated_at

    colnago = other_partitioned_db.update_record(bianchi, {"brand": "Colnago"})

    assert colnago.created_by == partitioned_db.user_id
    assert colnago.updated_by == other_partitioned_db.user_id
    assert colnago.created_at == bianchi.created_at
    assert colnago.updated_at > bianchi.updated_at

    colnago2 = partitioned_db.get_record(bianchi.id)

    assert colnago2.created_by == partitioned_db.user_id
    assert colnago2.updated_by == other_partitioned_db.user_id
    assert colnago2.created_at == colnago.created_at
    assert colnago2.updated_at == colnago.updated_at

    colnago3 = partitioned_db.get_record(bianchi.id)

    assert colnago3.created_by == partitioned_db.user_id
    assert colnago3.updated_by == other_partitioned_db.user_id
    assert colnago3.created_at == colnago.created_at
    assert colnago3.updated_at == colnago.updated_at


def test_delete_dataset(
    sample_patient_db, partitioned_db, another_partitioned_db, valid_organization, neo4j
):
    organization_id, organization_node_id = valid_organization

    bike = another_partitioned_db.create_model("bike", "Bicycle", "A bicycle")
    another_partitioned_db.update_properties(
        bike,
        ModelProperty(
            model_title=True,
            name="brand",
            display_name="Brand",
            data_type=dt.String(),
            description="Brand",
        ),
    )
    bianchi = another_partitioned_db.create_records(bike, [{"brand": "Bianchi"}])[0]
    another_partitioned_db.create_package_proxy(bianchi, 123, "N:package:123")
    stub = another_partitioned_db.create_model_relationship_stub("belongs_to")

    another_partitioned_db.delete_dataset()

    # Should delete the dataset node, but not the organization
    result = get_organization_and_dataset(another_partitioned_db)
    assert result is not None
    assert result["organization"]["id"] == organization_id.id
    assert result["organization"]["node_id"] == organization_node_id
    assert result["dataset"] is None

    # Should have no orphaned nodes
    assert len(get_orphaned_nodes(another_partitioned_db)) == 0

    # Should not touch the other dataset
    summary = partitioned_db.summarize()
    assert summary.model_count == 3
    assert summary.model_record_count == 7
    assert summary.relationship_count == 2
    assert summary.relationship_record_count == 5
    assert summary.relationship_type_count == 2


def test_search_permissions(
    sample_patient_db,
    partitioned_db,
    another_partitioned_db,
    valid_organization,
    valid_user,
):
    alice = sample_patient_db["records"]["alice"]
    patient = sample_patient_db["models"]["patient"]
    patient_properties = partitioned_db.get_properties(patient)
    dataset = Dataset(partitioned_db.dataset_id, partitioned_db.dataset_node_id)

    other_dataset = Dataset(
        another_partitioned_db.dataset_id, another_partitioned_db.dataset_node_id
    )

    # A "patient" model in another dataset
    other_patient = another_partitioned_db.create_model("patient", "Patient", "")
    other_patient_properties = another_partitioned_db.update_properties(
        other_patient,
        ModelProperty(
            name="name",
            display_name="Name",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
    )
    other_alice = another_partitioned_db.create_record(other_patient, {"name": "Alice"})

    # A "color" model in another dataset
    color = another_partitioned_db.create_model("color", "Color", "")
    another_partitioned_db.update_properties(
        color,
        ModelProperty(
            name="name",
            display_name="Name",
            data_type=dt.String(),
            description="",
            model_title=True,
        ),
    )
    red = another_partitioned_db.create_record(color, {"name": "red"})

    another_partitioned_db.create_model_relationship(other_patient, "likes", color)

    # No dataset permissions - should find nothing
    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[],
    )

    results, total_count = db.search_records(
        ModelFilter(name="patient"),
        property_filters=[
            PropertyFilter(model="patient", property_="name", value="Alice")
        ],
    )
    assert list(results) == []
    assert total_count == 0
    assert list(db.suggest_models()) == []

    # Permissions to one dataset - should find "Alice" in that dataset
    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(partitioned_db)],
    )
    results, total_count = db.search_records(
        ModelFilter(name="patient"),
        property_filters=[
            PropertyFilter(model="patient", property_="name", value="Alice")
        ],
    )
    assert list(results) == [
        SearchResult(patient.id, patient_properties, alice, dataset)
    ]
    assert [m for _, m in db.suggest_models()] == [
        ModelSuggestion("medication", "Medication"),
        ModelSuggestion("patient", "Patient"),
        ModelSuggestion("visit", "Visit"),
    ]
    assert [m for _, m in db.suggest_models(related_to="patient")] == [
        ModelSuggestion("patient", "Patient"),
        ModelSuggestion("visit", "Visit"),
    ]

    # Permissions to other dataset - should find "Alice" in that dataset
    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(another_partitioned_db)],
    )
    results, total_count = db.search_records(
        ModelFilter(name="patient"),
        property_filters=[
            PropertyFilter(model="patient", property_="name", value="Alice")
        ],
    )
    assert list(results) == [
        SearchResult(
            other_patient.id, other_patient_properties, other_alice, other_dataset
        )
    ]
    assert total_count == 1
    assert [m for _, m in db.suggest_models()] == [
        ModelSuggestion("color", "Color"),
        ModelSuggestion("patient", "Patient"),
    ]

    # Permissions to both datasets - should find "Alice" in both
    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[
            create_test_datasetDTO(partitioned_db),
            create_test_datasetDTO(another_partitioned_db),
        ],
    )
    results, total_count = db.search_records(
        ModelFilter(name="patient"),
        property_filters=[
            PropertyFilter(model="patient", property_="name", value="Alice")
        ],
    )
    assert sorted(results, key=lambda r: r.record.id) == sorted(
        [
            SearchResult(patient.id, patient_properties, alice, dataset),
            SearchResult(
                other_patient.id, other_patient_properties, other_alice, other_dataset
            ),
        ],
        key=lambda r: r.record.id,
    )

    assert list(
        unique_everseen((m for _, m in db.suggest_models()), key=lambda m: m.name)
    ) == [
        ModelSuggestion("color", "Color"),
        ModelSuggestion("medication", "Medication"),
        ModelSuggestion("patient", "Patient"),
        ModelSuggestion("visit", "Visit"),
    ]
    assert list(
        unique_everseen(
            (m for _, m in db.suggest_models(related_to="patient")),
            key=lambda m: m.name,
        )
    ) == [
        ModelSuggestion("color", "Color"),
        ModelSuggestion("patient", "Patient"),
        ModelSuggestion("visit", "Visit"),
    ]

    # Can search a specific dataset
    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[
            create_test_datasetDTO(partitioned_db),
            create_test_datasetDTO(another_partitioned_db),
        ],
    )
    results, total_count = db.search_records(
        ModelFilter(name="patient"),
        dataset_filters=[DatasetFilter(id=another_partitioned_db.dataset_id)],
    )
    assert list(results) == [
        SearchResult(
            other_patient.id, other_patient_properties, other_alice, other_dataset
        )
    ]


def test_search_is_case_insensitive(partitioned_db, movie_db):
    _, movie_db = movie_db

    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(partitioned_db)],
    )

    # Test case insensitivity
    # All lowercase
    assert (
        sort_by_record_name(
            db.search_records(
                ModelFilter(name="person"),
                [
                    PropertyFilter(
                        model="person", property_="name", value="carrie-anne moss"
                    )
                ],
            )
        )
        == [movie_db["Carrie"]]
    )

    # All uppercase
    assert (
        sort_by_record_name(
            db.search_records(
                ModelFilter(name="person"),
                [
                    PropertyFilter(
                        model="person", property_="name", value="CARRIE-ANNE MOSS"
                    )
                ],
            )
        )
        == [movie_db["Carrie"]]
    )

    # Mixed cases
    assert (
        sort_by_record_name(
            db.search_records(
                ModelFilter(name="person"),
                [
                    PropertyFilter(
                        model="person", property_="name", value="cArRiE-AnNe mOsS"
                    )
                ],
            )
        )
        == [movie_db["Carrie"]]
    )

    # Test wrong value type
    # Expects string, given number, should find nothing
    results, total_count = db.search_records(
        ModelFilter(name="person"),
        [PropertyFilter(model="person", property_="name", value=1971)],
    )

    assert list(results) == []
    assert total_count == 0

    # Expects number, given string, should find nothing
    results, total_count = db.search_records(
        ModelFilter(name="person"),
        [PropertyFilter(model="person", property_="born", value="carrie-anne moss")],
    )

    assert list(results) == []
    assert total_count == 0


def test_search_multiple_filters(partitioned_db, movie_db):
    _, movie_db = movie_db

    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(partitioned_db)],
    )

    # Raise an exception:
    with pytest.raises(ModelNotFoundError):
        db.suggest_properties(ModelFilter(name="nonsense"))

    assert sort_by_record_name(
        db.search_records(
            ModelFilter(name="person"),
            [PropertyFilter(model="person", property_="born", value=1967)],
        )
    ) == [
        movie_db["Carrie"],
        movie_db["JamesM"],
        movie_db["LillyW"],
        movie_db["SteveZ"],
    ]
    assert sorted(
        p.name for _, p, _ in db.suggest_properties(ModelFilter(name="person"))
    ) == ["born", "name"]

    assert (
        sort_by_record_name(
            db.search_records(
                ModelFilter(name="person"),
                [
                    PropertyFilter(model="person", property_="born", value=1967),
                    PropertyFilter(
                        model="person", property_="name", value="Carrie-Anne Moss"
                    ),
                ],
            )
        )
        == [movie_db["Carrie"]]
    )

    assert (
        sort_by_record_name(
            db.search_records(
                ModelFilter(name="person"),
                [
                    PropertyFilter(model="person", property_="born", value=1967),
                    PropertyFilter(
                        model="person", property_="name", value="James Marshall"
                    ),
                ],
            )
        )
        == [movie_db["JamesM"]]
    )


def test_search_one_hop(partitioned_db, movie_db):
    _, movie_db = movie_db

    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(partitioned_db)],
    )

    # Query basic relationships
    # (People related to "The Matrix")
    assert sort_by_record_name(
        db.search_records(
            ModelFilter(name="person"),
            [PropertyFilter(model="movie", property_="title", value="The Matrix")],
        )
    ) == [
        movie_db["Carrie"],
        movie_db["Hugo"],
        movie_db["JoelS"],
        movie_db["Keanu"],
        movie_db["LanaW"],
        movie_db["Laurence"],
        movie_db["LillyW"],
    ]

    # Can use operators other than equals
    # (People related to "The Matrix", "The Matrix Reloaded", "The Matrix Revolutions")
    assert sort_by_record_name(
        db.search_records(
            ModelFilter(name="person"),
            [
                PropertyFilter(
                    model="movie",
                    property_="title",
                    value="The Matrix",
                    operator=op.STARTS_WITH,
                )
            ],
        )
    ) == [
        movie_db["Carrie"],
        movie_db["Emil"],
        movie_db["Hugo"],
        movie_db["JoelS"],
        movie_db["Keanu"],
        movie_db["LanaW"],
        movie_db["Laurence"],
        movie_db["LillyW"],
    ]

    # Can apply multiple operators to the different properties on the same model
    # (People related to "The Matrix Revolutions")
    assert sort_by_record_name(
        db.search_records(
            ModelFilter(name="person"),
            [
                PropertyFilter(
                    model="movie",
                    property_="title",
                    value="The Matrix",
                    operator=op.STARTS_WITH,
                ),
                PropertyFilter(
                    model="movie",
                    property_="tag_line",
                    value="Everything that has a beginning has an end",
                    operator=op.EQUALS,
                ),
            ],
        )
    ) == [
        movie_db["Carrie"],
        movie_db["Hugo"],
        movie_db["JoelS"],
        movie_db["Keanu"],
        movie_db["LanaW"],
        movie_db["Laurence"],
        movie_db["LillyW"],
    ]

    # Can apply multiple operators to the different properties on the same model
    # (People related to "The Matrix Revolutions" AND who won an Academy Award)
    assert (
        sort_by_record_name(
            db.search_records(
                ModelFilter(name="person"),
                [
                    PropertyFilter(
                        model="movie",
                        property_="title",
                        value="The Matrix",
                        operator=op.STARTS_WITH,
                    ),
                    PropertyFilter(
                        model="award",
                        property_="name",
                        value="Academy Award",
                        operator=op.EQUALS,
                    ),
                ],
            )
        )
        == [movie_db["LanaW"], movie_db["LillyW"]]
    )

    # If a model is related to itself, and this model is the target of the
    # search, then all filters should be applied to the root record, not related
    # records.
    # Laurence, Aaron, Boonie and Meg were born in 1961
    knows = partitioned_db.create_model_relationship(
        movie_db["person"], "knows", movie_db["person"]
    )
    partitioned_db.create_record_relationship(
        movie_db["Laurence"], knows, movie_db["LillyW"]
    )
    partitioned_db.create_record_relationship(
        movie_db["Hugo"], knows, movie_db["AaronS"]
    )

    assert sort_by_record_name(
        db.search_records(
            ModelFilter(name="person"),
            [
                PropertyFilter(
                    model="person", property_="born", value=1961, operator=op.EQUALS
                )
            ],
        )
    ) == [
        movie_db["AaronS"],
        movie_db["BonnieH"],
        movie_db["Laurence"],
        movie_db["MegR"],
    ]


def test_suggest_values(movie_db, partitioned_db):
    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(partitioned_db)],
    )
    _, vs = movie_db

    # Boolean suggestion
    assert set(
        [p for _, p in db.suggest_values("imdb_page", "published")][0].values
    ) == set([True, False])

    # Long suggestion
    assert [p for _, p in db.suggest_values("person", "born")][0].values == [1929, 1996]

    # Date suggestion
    assert [p for _, p in db.suggest_values("movie", "date_of_release")][0].values == [
        datetime(1986, 5, 16, 0, 0, tzinfo=timezone.utc),
        datetime(2003, 11, 5, 0, 0, tzinfo=timezone.utc),
    ]

    # String suggestion (with prefix)
    assert [p for _, p in db.suggest_values("movie", "tag_line", matching_prefix="ev")][
        0
    ].values == [
        "Everything that has a beginning has an end",
        "Evil has its winning ways",
    ]

    # String suggestion (without prefix)
    assert len([p for _, p in db.suggest_values("movie", "tag_line")][0].values) == 10

    # Enumeration suggestion
    assert set([p for _, p in db.suggest_values("movie", "rating")][0].values) == set(
        ["Unwatchable", "Poor", "Fair", "Good", "Excellent"]
    )

    # Array suggestion (no prefix)
    assert set([p for _, p in db.suggest_values("movie", "tags")][0].values) == set(
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

    # Array suggestion (prefix)
    assert set(
        [p for _, p in db.suggest_values("movie", "tags", matching_prefix="s")][
            0
        ].values
    ) == set(["scifi", "sports", "summer", "spoon", "seattle"])

    array_model = partitioned_db.create_model("array_model", "Array Model")
    partitioned_db.update_properties(
        array_model,
        ModelProperty(
            name="array_numbers",
            display_name="Array Numbers",
            data_type=dt.Array(items=dt.Long()),
            description="",
            model_title=True,
        ),
        ModelProperty(
            name="array_dates",
            display_name="Array Dates",
            data_type=dt.Array(items=dt.Date()),
            description="",
            model_title=False,
        ),
    )
    partitioned_db.create_records(
        array_model,
        [
            {
                "array_numbers": [1, 2, 3],
                "array_dates": [
                    datetime(1986, 5, 16, 0, 0, tzinfo=timezone.utc),
                    datetime(2003, 11, 5, 0, 0, tzinfo=timezone.utc),
                ],
            },
            {
                "array_numbers": [3, 6, 7],
                "array_dates": [datetime(1986, 5, 16, 0, 0, tzinfo=timezone.utc)],
            },
        ],
    )

    # Array suggestion (numeric)
    assert set(
        [p for _, p in db.suggest_values("array_model", "array_numbers")][0].values
    ) == set([3, 1, 2, 6, 7])

    # Array suggestion (date)
    assert set(
        [
            normalize_datetime(d)
            for d in (
                [d for _, d in db.suggest_values("array_model", "array_dates")][
                    0
                ].values
            )
        ]
    ) == set(
        [
            datetime(1986, 5, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2003, 11, 5, 0, 0, tzinfo=timezone.utc),
        ]
    )


def test_search_limit_offset_total_count(partitioned_db, movie_db):
    _, movie_db = movie_db

    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(partitioned_db)],
    )

    # Query basic relationships
    # (People related to "The Matrix")

    records, total_count = db.search_records(
        ModelFilter(name="person"),
        [PropertyFilter(model="movie", property_="title", value="The Matrix")],
    )
    records = list(records)
    assert total_count == 7
    assert len(records) == 7

    records1, total_count = db.search_records(
        ModelFilter(name="person"),
        [PropertyFilter(model="movie", property_="title", value="The Matrix")],
        limit=2,
        offset=3,
    )
    records1 = list(records1)
    assert total_count == 7
    assert records1 == records[3:5]


def test_search_packages(partitioned_db, movie_db):
    _, movie_db = movie_db

    db = SearchDatabase(
        partitioned_db.db,
        partitioned_db.organization_id,
        partitioned_db.user_id,
        datasets=[create_test_datasetDTO(partitioned_db)],
    )

    dataset = Dataset(partitioned_db.dataset_id, partitioned_db.dataset_node_id)

    pp1 = partitioned_db.create_package_proxy(
        movie_db["Hugo"].id, package_id=1234, package_node_id="N:dataset:1234"
    )
    package_Hugo = Package(pp1.package_id, pp1.package_node_id)

    pp2 = partitioned_db.create_package_proxy(
        movie_db["LillyW"].id, package_id=5678, package_node_id="N:dataset:5678"
    )
    package_LillyW = Package(pp2.package_id, pp2.package_node_id)

    # Can use operators other than equals
    # (People related to "The Matrix", "The Matrix Reloaded", "The Matrix Revolutions")
    packages, total_count = db.search_packages(
        ModelFilter(name="person"),
        [
            PropertyFilter(
                model="movie",
                property_="title",
                value="The Matrix",
                operator=op.STARTS_WITH,
            )
        ],
    )
    assert total_count == 2
    assert list(packages) == [
        PackageSearchResult(package_Hugo, dataset),
        PackageSearchResult(package_LillyW, dataset),
    ]

    # Can apply multiple operators to the different properties on the same model
    # (People related to "The Matrix Revolutions" AND who won an Academy Award)
    packages, total_count = db.search_packages(
        ModelFilter(name="person"),
        [
            PropertyFilter(
                model="movie",
                property_="title",
                value="The Matrix",
                operator=op.STARTS_WITH,
            ),
            PropertyFilter(
                model="award",
                property_="name",
                value="Academy Award",
                operator=op.EQUALS,
            ),
        ],
    )
    assert total_count == 1
    assert list(packages) == [PackageSearchResult(package_LillyW, dataset)]


def sort_by_record_name(results: Tuple[Iterator[SearchResult], int]) -> List[Record]:
    return sorted(list(r.record for r in results[0]), key=lambda r: r.values["name"])


def test_duplicate_related_to_model_relationships(partitioned_db):
    """
    Ensure that all model relationship information is tracked on the
    `@RELATED_TO` relationship.
    """
    patient = partitioned_db.create_model("patient", "Patient", "")
    visit = partitioned_db.create_model("visit", "Visit", "")
    assert get_related_to_model_relationship(partitioned_db, patient, visit) == []

    attends1 = partitioned_db.create_model_relationship(patient, "attends", visit)
    assert get_related_to_model_relationship(partitioned_db, patient, visit) == [
        attends1
    ]

    attends2 = partitioned_db.update_model_relationship(
        attends1, display_name="Present", index=100
    )
    assert attends2.display_name == "Present"
    assert attends2.index == 100
    assert get_related_to_model_relationship(partitioned_db, patient, visit) == [
        attends2
    ]

    with partitioned_db.transaction() as tx:
        partitioned_db.delete_model_relationship_tx(tx=tx, relationship=attends1)
    assert get_related_to_model_relationship(partitioned_db, patient, visit) == []


def get_related_to_model_relationship(db, from_, to):
    """
    Retrieve the duplicate `@RELATED_TO` relationship between two models.
    """
    return [
        ModelRelationship(
            id=node["r"]["id"],
            type=node["r"]["type"],
            name=node["r"]["name"],
            from_=node["from"],
            to=node["to"],
            one_to_many=node["r"]["one_to_many"],
            display_name=node["r"]["display_name"],
            description=node["r"]["description"],
            created_at=node["r"]["created_at"],
            updated_at=node["r"]["updated_at"],
            created_by=node["r"]["created_by"],
            updated_by=node["r"]["updated_by"],
            index=node["r"]["index"],
        )
        for node in db.execute_single(
            f"""
            MATCH (m {{ id: $from_id }})-[{labels.related_to("r")}]->(n {{ id: $to_id }})
            RETURN r, m.id AS from, n.id AS to
            """,
            from_id=from_.id,
            to_id=to.id,
        ).records()
    ]


def test_assert_model_relationship_exists(sample_patient_db, partitioned_db):

    alice = sample_patient_db["records"]["alice"]
    tuesday = sample_patient_db["records"]["tuesday"]
    aspirin = sample_patient_db["records"]["aspirin"]

    with partitioned_db.transaction() as tx:
        model_relationship = partitioned_db.assert_.model_relationship_exists(
            tx, alice, "attends", tuesday
        )

        assert model_relationship.from_ == sample_patient_db["models"]["patient"].id
        assert model_relationship.to == sample_patient_db["models"]["visit"].id
        assert model_relationship.name == "attends"
        assert model_relationship.type == "ATTENDS"

    with pytest.raises(ModelRelationshipNotFoundError):
        with partitioned_db.transaction() as tx:
            partitioned_db.assert_.model_relationship_exists(
                tx, alice, "attends", aspirin
            )


def test_delete_large_dataset_rolled_back_for_large_batch_size(large_partitioned_db):
    # `large_partitioned_db` has 15k Record nodes
    before_summary = large_partitioned_db.summarize()
    assert before_summary.model_count > 0
    assert before_summary.model_record_count > 0

    # If we choose a big batch size for the initial call with a short duration,
    # no records will be removed (and subsequently rolled back). We will also
    # get a time limit exceeded error:
    with pytest.raises(ExceededTimeLimitError):
        large_partitioned_db.delete_dataset(batch_size=10000, duration=100)

    # If we drop the batch size, the call will succeed
    result = large_partitioned_db.delete_dataset(batch_size=1000, duration=500)
    assert not result.done
    assert result.counts.records >= 1000


def test_delete_large_dataset_with_time_limit(
    large_partitioned_db, another_partitioned_db
):
    before_summary = large_partitioned_db.summarize()
    assert before_summary.model_count > 0
    assert before_summary.model_record_count > 0

    deleted_models, deleted_records = 0, 0
    batch_size = 1000
    duration = 500

    # For 15k nodes, it should take a few runs to actually delete all nodes,
    # if a maximum run time of 500ms is allowed. This allows for multiple calls
    # to `delete_dataset` to actually clear all dependent nodes from a dataset:
    result = large_partitioned_db.delete_dataset(
        batch_size=batch_size, duration=duration
    )
    assert not result.done
    assert result.counts.records > 0
    assert result.counts.models == 0
    deleted_models += result.counts.models
    deleted_records += result.counts.records

    # Loop until everything is cleared out:
    for _ in range(100):
        result = large_partitioned_db.delete_dataset(
            batch_size=batch_size, duration=duration
        )
        deleted_models += result.counts.models
        deleted_records += result.counts.records
        if result.done:
            break
    else:
        assert result.done  # will fail otherwise

    # The number of deleted models should be the same as the number from
    # the starting summary:
    assert result.done
    assert before_summary.model_count == deleted_models
    assert before_summary.model_record_count == deleted_records

    after_summary = another_partitioned_db.summarize()
    assert after_summary.model_count == 0
    assert after_summary.model_record_count == 0
    assert after_summary.relationship_count == 0
    assert after_summary.relationship_record_count == 0
    assert after_summary.relationship_type_count == 0
