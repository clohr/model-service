from typing import List, Optional, Tuple, Union

import connexion
from auth_middleware.models import DatasetPermission  # type: ignore
from werkzeug.exceptions import NotFound

from server.auth import permission_required
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.models import (
    CreateRecordRelationship,
    JsonDict,
    ModelRelationshipId,
    RecordRelationshipId,
)
from server.models.legacy import CreateModelRelationship

from ...errors import OperationError
from .common import (
    to_legacy_relationship,
    to_legacy_relationship_instance,
    to_relationship_id_or_name,
)

RelationshipInstancePair = Tuple[JsonDict, JsonDict]


# NOTE: We don't require the model ID to fetch a record if given the
# record's ID
@permission_required(DatasetPermission.VIEW_RECORDS)
def get_concept_instance_relationship(
    db: PartitionedDatabase,
    relationship_id_or_name: ModelRelationshipId,  # unused
    id_: RecordRelationshipId,
):
    with db.transaction() as tx:
        return to_legacy_relationship_instance(
            db.get_record_relationship_tx(tx, relationship_id_or_name, id_)
        )


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_concept_instance_relationships(
    db: PartitionedDatabase, relationship_id_or_name: ModelRelationshipId
) -> List[JsonDict]:
    with db.transaction() as tx:
        try:
            stub = db.get_model_relationship_stub_tx(
                tx=tx, relation=relationship_id_or_name
            )
        except OperationError:
            stub = None

        if stub == None:
            return [
                to_legacy_relationship_instance(r)
                for r in db.get_record_relationships_by_model_tx(
                    tx, relationship_id_or_name
                )
            ]
        else:
            return [
                to_legacy_relationship_instance(r)
                for r in db.get_record_relationships_by_model_with_relationship_stub_tx(
                    tx, stub
                )
            ]


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def create_concept_instance_relationships(
    db: PartitionedDatabase,
    relationship_id_or_name: Union[ModelRelationshipId, str],
    body: List[JsonDict],
) -> List[RelationshipInstancePair]:

    relationship_to_create_schema = CreateModelRelationship.schema()
    to_create = []

    for row in body:
        if row.get("relationshipToCreate"):
            model_relationship_to_create = relationship_to_create_schema.load(
                row["relationshipToCreate"]
            )
        else:
            model_relationship_to_create = None

        to_create.append(
            CreateRecordRelationship(
                from_=row["from"],
                to=row["to"],
                model_relationship_to_create=model_relationship_to_create,
            )
        )

    with db.transaction() as tx:
        rels = db.create_legacy_record_relationship_batch_tx(
            tx=tx, to_create=to_create, relation=relationship_id_or_name
        )
    return [
        (
            to_legacy_relationship_instance(record_relationship),
            to_legacy_relationship(model_relationship),
        )
        for record_relationship, model_relationship in rels
    ]


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def create_concept_instance_relationship(
    db: PartitionedDatabase,
    relationship_id_or_name: Union[ModelRelationshipId, str],
    body: JsonDict,
) -> Tuple[RelationshipInstancePair, int]:
    relationship_to_create_schema = CreateModelRelationship.schema()
    model_relationship_to_create: Optional[CreateModelRelationship] = None

    if body.get("relationshipToCreate"):
        model_relationship_to_create = relationship_to_create_schema.load(
            row["relationshipToCreate"]
        )

    to_create = CreateRecordRelationship(
        from_=body.get("from"),
        to=body.get("to"),
        model_relationship_to_create=model_relationship_to_create,
    )

    with db.transaction() as tx:
        rels = db.create_legacy_record_relationship_batch_tx(
            tx=tx, to_create=[to_create], relation=relationship_id_or_name
        )

        return (
            (
                to_legacy_relationship_instance(rels[0][0]),
                to_legacy_relationship(rels[0][1]),
            ),
            200,
        )


# NOTE: We don't require the model relationship ID (`relationship_id`)
# to delete a record if given the relationship ID.
@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def delete_concept_instance_relationship(
    db: PartitionedDatabase,
    relationship_id_or_name: ModelRelationshipId,  # ununsed
    id_: RecordRelationshipId,
) -> RecordRelationshipId:

    with db.transaction() as tx:
        deleted = db.delete_record_relationships_tx(tx, id_)

    if deleted is None or len(deleted) == 0:
        raise NotFound(f"Could not delete record relationship [{id_}]")
    return deleted[0]


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def delete_concept_instance_relationships(db: PartitionedDatabase) -> List[JsonDict]:

    # HACK: request bodies on DELETE requests do not have defined
    # semantics and are not directly support by OpenAPI/Connexion. See
    #  - https://swagger.io/docs/specification/describing-request-body
    #  - https://github.com/zalando/connexion/issues/896
    body = connexion.request.json

    # HACK:
    # since we're pulling directly from the raw body, names will not be camel cased:
    relationship_instance_ids: List[str] = body.get("relationshipInstanceIds", [])

    with db.transaction() as tx:
        return db.delete_record_relationships_tx(tx, *relationship_instance_ids)
