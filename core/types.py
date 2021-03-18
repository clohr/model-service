from datetime import datetime
from typing import Any, Dict, List, NewType, Optional, Tuple, Union
from uuid import UUID

OrganizationId = NewType("OrganizationId", int)
OrganizationNodeId = NewType("OrganizationNodeId", str)  # "N:organization:<uuidv4>"
UserId = NewType("UserId", int)
UserNodeId = NewType("UserNodeId", str)  # "N:user:<uuidv4>"
DatasetId = NewType("DatasetId", int)
DatasetNodeId = NewType("DatasetNodeId", str)  # "N:dataset:<uuidv4>"
ModelId = NewType("ModelId", UUID)
ModelRelationshipId = NewType("ModelRelationshipId", UUID)
RecordId = NewType("RecordId", UUID)
RecordRelationshipId = NewType("RecordRelationshipId", UUID)
RelationshipName = NewType("RelationshipName", str)
RelationshipType = NewType("RelationshipType", str)
ModelPropertyId = NewType("ModelPropertyId", UUID)
PackageId = NewType("PackageId", int)
PackageNodeId = NewType("PackageNodeId", str)  # "N:package:<uuidv4>"
PackageProxyId = NewType("PackageProxyId", UUID)

NativeScalar = Union[bool, int, str, float, datetime]

# TODO Recursive types aren't support currently (https://github.com/python/mypy/issues/7069)
GraphValue = Optional[
    Union[
        bool,
        List[bool],
        int,
        List[int],
        float,
        List[float],
        str,
        List[str],
        datetime,
        List[datetime],
        # List["GraphValue"] - not supported
        List[object],
        # Dict[str, "GraphValue"] - not supported
        Dict[str, object],
        List[Dict[str, object]],
        OrganizationId,
        DatasetId,
        List[DatasetId],
        RecordId,
        List[RecordId],
        ModelId,
        List[ModelId],
        ModelRelationshipId,
        List[ModelRelationshipId],
        RecordRelationshipId,
        List[RecordRelationshipId],
        ModelPropertyId,
        List[ModelPropertyId],
        PackageId,
        List[PackageId],
        PackageProxyId,
        List[PackageProxyId],
        UUID,
        List[UUID],
    ]
]
JsonDict = Dict[str, GraphValue]
NextPageCursor = NewType("NextPageCursor", int)
