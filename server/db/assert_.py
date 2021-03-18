from typing import Callable, Dict, Union

from neo4j import Session, Transaction  # type: ignore

from ..errors import (
    DuplicateModelNameError,
    DuplicateModelPropertyNameError,
    DuplicateModelRelationshipError,
    ModelInUseError,
    ModelNameCountError,
    ModelNotFoundError,
    ModelPropertyInUseError,
    ModelRelationshipNotFoundError,
    MultiplePropertyDisplayNameError,
    MultiplePropertyNameError,
    MultiplePropertyTitleError,
    MultipleRelationshipsViolationError,
    NoPropertyTitleError,
    RecordNotFoundError,
)
from ..models import (
    GraphValue,
    Model,
    ModelId,
    ModelProperty,
    ModelPropertyId,
    ModelRelationship,
    ModelRelationshipId,
    Record,
    RecordId,
    RelationshipName,
    get_record_id,
    get_relationship_type,
)
from . import core, labels
from .util import match_clause


class AssertionHelper:
    def __init__(self, db: "core.PartitionedDatabase"):
        self._db = db

    def model_count_check(
        self,
        tx: Union[Session, Transaction],
        name: str,
        predicate: Callable[[int], bool],
    ) -> None:
        """
        Check that the count associated with given model name fulfills some
        predicate.

        Note: `tx` can be any object that supplies a `run()` method.

        Raises
        ------
        ModelNameCountError
        """

        cql = f"""
        MATCH  ({labels.model("m")} {{ name: $name }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        RETURN COUNT(m) AS count
        """

        count = (
            tx.run(
                cql,
                organization_id=self._db.organization_id,
                dataset_id=self._db.dataset_id,
                name=name,
            )
            .single()
            .get("count")
        )

        if not predicate(count):
            raise ModelNameCountError(model_name=name, count=count)

    def model_name_does_not_exist(
        self, tx: Union[Session, Transaction], name: str
    ) -> None:
        """
        Check that the given model name does not yet exist in the current
        (organization, dataset).

        Note: `tx` can be any object that supplies a `run()` method.
        Raises
        ------
        DuplicateModelNameError
        """
        try:
            self.model_count_check(tx=tx, name=name, predicate=lambda count: count == 0)
        except ModelNameCountError:
            raise DuplicateModelNameError(name)

    def model_name_is_unique(self, tx: Union[Session, Transaction], name: str) -> None:
        """
        Check that the given model name is unique (at most 1 instance
        currently exists) for the current (organization, dataset).

        Note: `tx` can be any object that supplies a `run()` method.
        Raises
        ------
        DuplicateModelNameError
        """
        try:
            self.model_count_check(tx=tx, name=name, predicate=lambda count: count <= 1)
        except ModelNameCountError:
            raise DuplicateModelNameError(name)

    def property_name_is_unique(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, ModelId, str],
        property_name: str,
    ) -> None:
        """
        Check that the given property name is unique for a model.

        Note: `tx` can be any object that supplies a `run()` method.

        Raises
        ------
        DuplicateModelPropertyNameError
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self._db.organization_id,
            dataset=self._db.dataset_id,
            property_name=property_name,
        )
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        MATCH (m)-[{labels.has_property()}]->(labels.model_property("p") {{ name: $property_name }})
        RETURN COUNT(p) AS count
        """
        count = tx.run(cql, **kwargs).single().get("count")

        if count > 0:
            raise DuplicateModelPropertyNameError(str(model_id_or_name), property_name)

    def model_relationship_exists(
        self,
        tx: Union[Session, Transaction],
        from_record: Union[Record, RecordId],
        relationship: Union[ModelRelationship, ModelRelationshipId, RelationshipName],
        to_record: Union[Record, RecordId],
    ) -> ModelRelationship:
        """
        Given two records, test that a relationship exists for the two
        corresponding models at the schema level.

        Note: `tx` can be any object that supplies a `run()` method.

        Raises
        ------
        ModelRelationshipNotFoundError
        """
        if isinstance(relationship, ModelRelationship):
            relationship = relationship.id

        check_constraint_cql = f"""
        MATCH ({labels.record("r1")} {{ `@id`: $from_record }})
              -[{labels.instance_of()}]->({labels.model("m1")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        MATCH ({labels.record("r2")} {{ `@id`: $to_record }})
              -[{labels.instance_of()}]->({labels.model("m2")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        MATCH (m1)-[{labels.related_to("rel")} {{ {match_clause("relationship", relationship)} }}]->(m2)

        RETURN r1.id           AS record_from,
               r2.id           AS record_to,
               m1.id           AS model_from,
               m2.id           AS model_to,
               rel
        """
        from_record = get_record_id(from_record)
        to_record = get_record_id(to_record)

        r = tx.run(
            check_constraint_cql,
            organization_id=self._db.organization_id,
            dataset_id=self._db.dataset_id,
            from_record=str(from_record),
            to_record=str(to_record),
            relationship=relationship,
        ).single()

        if not r:
            raise ModelRelationshipNotFoundError(
                id=str(relationship), model_from=str("*"), model_to=str("*"), type=None
            )

        return ModelRelationship(
            id=r["rel"]["id"],
            type=r["rel"]["type"],
            name=r["rel"]["name"],
            from_=r["model_from"],
            to=r["model_to"],
            one_to_many=r["rel"]["one_to_many"],
            display_name=r["rel"]["display_name"],
            description=r["rel"]["description"],
            created_by=r["rel"]["created_by"],
            updated_by=r["rel"]["updated_by"],
            created_at=r["rel"]["created_at"],
            updated_at=r["rel"]["updated_at"],
            index=r["rel"]["index"],
        )

    def model_relationship_name_is_unique(
        self,
        tx: Union[Session, Transaction],
        relationship_name: Union[RelationshipName, str],
        from_model: Union[Model, ModelId],
        to_model: Union[Model, ModelId],
    ) -> None:
        """
        Assert that no relationship (count == 0) with the given name exists
        going from `from_model` to the `to_model`.

        Ideally this would be more strict and check that the relationship type
        between two models is unique, but I do not think that we can guarantee
        that for old data imported from `concepts-service`.
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self._db.organization_id,
            dataset_id=self._db.dataset_id,
            relationship_name=relationship_name,
        )

        cql = f"""
        MATCH  ({labels.model("m1")} {{ {match_clause("from_model", from_model, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.model("m2")} {{ {match_clause("to_model", to_model, kwargs)} }})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        MATCH (m1)-[{labels.related_to("relationship")} {{ name: $relationship_name }}]->(m2)

        RETURN COUNT(relationship) AS count
        """
        count = tx.run(cql, **kwargs).single().get("count", 0)

        if count > 0:
            raise DuplicateModelRelationshipError(
                relationship_name, from_model, to_model
            )

        return None

    def one_or_many_condition_holds(
        self,
        tx: Union[Session, Transaction],
        from_record: Union[Record, RecordId],
        model_relationship: ModelRelationship,
    ) -> None:
        """
        Assert that if `model_relationship.one_to_many = False`, any record
        relationships that are an instance of `model_relationship` must have
        exactly zero or one instances.

        Note: `tx` can be any object that supplies a `run()` method.

        Raises
        ------
        MultipleRelationshipsViolationError
        """
        from_record = get_record_id(from_record)
        rel_id = ModelRelationshipId(model_relationship.id)

        if not model_relationship.one_to_many:
            count = self._db._count_outgoing_records_tx(tx, from_record, rel_id)
            if count > 1:
                raise MultipleRelationshipsViolationError(
                    from_record, rel_id, model_relationship.type
                )

    def single_model_title(self, tx: Union[Session, Transaction], model: Model) -> None:
        """
        Assert that at most 1 property has `model_title=True` set for a given
        model.

        Raises
        ------
        MultiplePropertyTitleError
        """
        titles = [p for p in self._db.get_properties_tx(tx, model) if p.model_title]

        if len(titles) > 1:
            raise MultiplePropertyTitleError(str(model), [p.name for p in titles])

        if len(titles) < 1:
            raise NoPropertyTitleError(str(model))

    def unique_property_names(
        self, tx: Union[Session, Transaction], model: Model
    ) -> None:

        cql = f"""
        MATCH ({labels.model_property("p")})
             <-[{labels.has_property()}]-({labels.model("m")} {{ id: $model_id }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH p.name AS property_name, COUNT(p) as count
        WHERE count > 1

        RETURN COLLECT(property_name) AS duplicate_names
        """
        node = tx.run(
            cql,
            model_id=model.id,
            organization_id=self._db.organization_id,
            dataset_id=self._db.dataset_id,
        ).single()

        if node and len(node["duplicate_names"]) > 0:
            # TODO: return all duplicates
            raise MultiplePropertyNameError(model.name, node["duplicate_names"][0])

    def unique_property_display_names(
        self, tx: Union[Session, Transaction], model: Model
    ) -> None:
        cql = f"""
        MATCH ({labels.model_property("p")})
             <-[{labels.has_property()}]-({labels.model("m")} {{ id: $model_id }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH p.display_name AS property_display_name, COUNT(p) as count
        WHERE count > 1

        RETURN COLLECT(property_display_name) AS duplicate_display_names
        """
        node = tx.run(
            cql,
            model_id=model.id,
            organization_id=self._db.organization_id,
            dataset_id=self._db.dataset_id,
        ).single()

        if node and len(node["duplicate_display_names"]) > 0:
            # TODO: return all duplicates
            raise MultiplePropertyDisplayNameError(
                model.name, node["duplicate_display_names"][0]
            )

    def model_property_not_used(
        self,
        tx: Union[Session, Transaction],
        model: Union[Model, ModelId, str],
        property_: Union[ModelProperty, ModelPropertyId, str],
    ) -> None:
        """
        Assert that no records reference the given model property.

        Raises
        ------
        ModelPropertyInUseError
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self._db.organization_id, dataset_id=self._db.dataset_id
        )
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_id_or_name", model, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        OPTIONAL MATCH  ({labels.record("r")})
                       -[{labels.instance_of()}]->(m)
                       -[{labels.has_property()}]->({labels.model_property("p")} {{ {match_clause("property_id_or_name", property_, kwargs)} }})
        WHERE r[p.name] IS NOT NULL
        RETURN COUNT(r) AS property_count,
               p.name   AS property_name
        """
        node = tx.run(cql, **kwargs).single()
        count = node["property_count"]
        prop_name = node["property_name"] or "<empty>"

        if count > 0:
            raise ModelPropertyInUseError(str(model), prop_name, usage_count=count)

    def record_exists(
        self, tx: Union[Session, Transaction], record: Union[Record, RecordId, str]
    ) -> None:
        """
        Raises
        ------
        RecordNotFoundError
        """

        record_id = get_record_id(record)

        cql = f"""
        MATCH  ({labels.record("r")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        RETURN r.id AS id
        """
        node = tx.run(
            cql,
            organization_id=self._db.organization_id,
            dataset_id=self._db.dataset_id,
            record_id=str(record_id),
        ).single()

        if node is None:
            raise RecordNotFoundError(record_id)

    def model_has_no_records(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, ModelId, str],
    ) -> None:
        """
        Raises
        ------
        ModelInUseError
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self._db.organization_id, dataset_id=self._db.dataset_id
        )

        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        OPTIONAL MATCH ({labels.record("r")})
                       -[{labels.instance_of()}]->(m)
        RETURN COUNT(r) > 0 AS model_in_use
        """
        node = tx.run(cql, **kwargs).single()

        if node is None:
            raise ModelNotFoundError(model_id_or_name)

        if node["model_in_use"]:
            raise ModelInUseError(model_id_or_name)
