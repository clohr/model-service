import json
from collections import Mapping, Set

import humps  # type: ignore
from marshmallow import Schema, pre_dump  # type: ignore


def _drop_nulls(o):
    """
    Drop `None` valued keys from an object.
    """
    if isinstance(o, (dict, Mapping)):
        return {k: _drop_nulls(v) for k, v in o.items() if v is not None}
    elif isinstance(o, list):
        return [_drop_nulls(v) for v in o if v is not None]
    elif isinstance(o, set):
        return {_drop_nulls(v) for v in o if v is not None}
    else:
        return o


class CamelCaseSchema(Schema):
    """Schema that uses camel-case for its external representation
    and snake-case for its internal representation.
    """

    def on_bind_field(self, field_name: str, field_obj) -> None:
        field_obj.data_key = humps.camelize(field_obj.data_key or field_name)

    class Meta:
        ordered = True


class Serializable:
    """
    Marks that a class is serializeable to JSON.
    """

    @classmethod
    def schema(cls):
        """
        Gets the marshmallow serializer for the implementing class.
        """
        schema = getattr(cls, "__schema__")
        if schema is None:
            raise Exception(f"{cls.__name__}: not serializable; missing schema")
        return schema

    def to_json(
        self,
        camel_case: bool = True,
        pretty_print: bool = True,
        drop_nulls: bool = False,
    ) -> str:
        """
        Convert an implementing instance to JSON.

        Parameters
        ----------
        camel_case : bool (default True)
            If True, the keys of the returned dict will be camel-cased.

        pretty_print : bool (default True)
            If True, JSON will be formatted prior to being returned.

        drop_nulls : bool (default False)
            If True, keys with values of `None` will be dropped from the
            returned JSON output.
        """
        d = (
            self.schema().dump(self)
            if camel_case
            else {humps.decamelize(k): v for k, v in self.schema().dump(self).items()}
        )
        if drop_nulls:
            d = _drop_nulls(d)
        s = json.dumps(d, indent=4 if pretty_print else None)
        return s

    def to_dict(self, camel_case: bool = True):
        """
        Convert an implementing instance to a Python dictionary.

        Parameters
        ----------
        camel_case : bool
            If true, the keys of the returned dict will be camel-cased.
        """
        if camel_case:  # camel case is used by default
            return self.schema().dump(self)
        return {humps.decamelize(k): v for k, v in self.schema().dump(self).items()}
