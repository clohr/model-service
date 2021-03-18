from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Set

import marshmallow  # type: ignore
from marshmallow import fields, post_load  # type: ignore

from ..json import CamelCaseSchema, Serializable
from ..util import normalize_datetime


class DatasetSchema(CamelCaseSchema):
    id = fields.String()
    int_id = fields.Integer()
    name = fields.String()

    @post_load
    def make(self, data, **kwargs):
        return Dataset(**data)


@dataclass(frozen=True)
class Dataset(Serializable):
    """
    A representation of a Dataset on the Pennsieve data platform.
    """

    __schema__: ClassVar[DatasetSchema] = DatasetSchema(unknown=marshmallow.EXCLUDE)

    PUBLIC: ClassVar[Set[str]] = set(["id", "int_id", "name"])

    id: str
    int_id: int
    name: str
