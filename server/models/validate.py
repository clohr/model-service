import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..errors import (
    InvalidPropertyDefaultValue,
    ModelNameEmptyError,
    ModelNameTooLongError,
    ModelNameValidationError,
    MultiplePropertyDisplayNameError,
    MultiplePropertyNameError,
    PropertyNameEmptyError,
    PropertyNameTooLongError,
    PropertyNameValidationError,
    RecordValidationError,
    RelationshipValidationError,
)
from . import GraphValue, RelationshipName
from .models import ModelProperty
from .util import RESERVED_PREFIX


@dataclass
class Result:
    errors: List[Tuple[str, Exception]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """
        Test if a validation error has occurred.
        """
        return len(self.errors) == 0

    def check(self) -> None:
        """
        Check if any errors have occurred, and if so, raise an error

        Raises
        ------
        RecordValidationError
        """
        if not self.ok:
            raise RecordValidationError(with_errors=self.errors)

    def error(self, property_name: str) -> Optional[Exception]:
        """
        Get the error on the named property.
        """
        for (prop_name, err) in self.errors:
            if prop_name == property_name:
                return err
        return None


# Will match all unicode letters, but not digits, and underscore "_",
# followed by letters, digits, and underscores:
MODEL_AND_PROPERTY_NAME_REGEX = re.compile(r"^([^\W\d_\-]|_)[\w_]*$")


def _validate_model_or_property_name(name: str, is_model: bool) -> str:
    name = name.strip()
    n = len(name)
    emptyError, nameTooLongError, validationError = (
        (ModelNameEmptyError, ModelNameTooLongError, ModelNameValidationError)
        if is_model
        else (
            PropertyNameEmptyError,
            PropertyNameTooLongError,
            PropertyNameValidationError,
        )
    )
    if n == 0:
        raise emptyError
    elif n > 64:
        raise nameTooLongError(name)
    if MODEL_AND_PROPERTY_NAME_REGEX.match(name) is None:
        raise validationError(name)
    return name


def validate_model_name(name: str) -> str:
    """
    Model names must:

    - be non-empty
    - be less than 64 characters
    - begin with a letter or underscore and consist of letters, digits, or
      underscores thereafter

    Raises
    ------
    - ModelNameEmptyError
    - ModelNameTooLongError
    - ModelNameValidationError
    """
    return _validate_model_or_property_name(name, is_model=True)


def validate_property_name(name: str) -> str:
    """
    Property names must:

    - be non-empty
    - be less than 64 characters
    - begin with a letter or underscore and consist of letters, digits, or
      underscores thereafter

    Raises
    ------
    - PropertyNameEmptyError
    - PropertyNameTooLongError
    - PropertyNameValidationError
    """
    return _validate_model_or_property_name(name, is_model=False)


RELATIONSHIP_NAME_REGEX = re.compile(r"^[a-zA-Z0-9][._\-a-zA-Z0-9/]*$")


def validate_relationship_name(name: str) -> RelationshipName:
    """
    Relationship names may only contain ASCII letters, numbers, and underscores,
    slashes (/) and may start with an underscore.

    Raises
    ------
    RelationshipValidationError
    """
    if RELATIONSHIP_NAME_REGEX.match(name) is None:
        raise RelationshipValidationError(relationship_name=name)
    return RelationshipName(name)


def validate_default_value(property_: ModelProperty):
    """
    If a default value is provided with `property_`, validate that a value
    is of the data type supplied with `property_`.
    """
    if property_.default:
        default_value: Optional[GraphValue] = property_.default_value
        if default_value is not None:
            try:
                property_.data_type.validate(default_value)
            except:  # noqa: E772
                raise InvalidPropertyDefaultValue(property_.name, default_value)


def validate_records(
    properties: List[ModelProperty], *records: Dict[str, GraphValue]
) -> Result:
    """
    Validate record values according to their datatypes.
    """
    # Build a map of the properties and collect required properties:
    property_map: Dict[str, ModelProperty] = {p.name: p for p in properties}

    # Collect required names
    required: Set[str] = set(
        p.name for p in property_map.values() if p.required or p.model_title
    )

    errors = []

    for i, record in enumerate(records, start=1):

        # Check for null values for required properties:
        for required_prop in required:
            if required_prop not in record:
                error_val = (
                    "model title"
                    if property_map[required_prop].model_title
                    else "required"
                )
                errors.append(
                    (
                        required_prop,
                        Exception(
                            f'validate_records@[{i}]: missing {error_val} property "{required_prop}"'
                        ),
                    )
                )
                continue
            if record[required_prop] is None:
                errors.append(
                    (
                        required_prop,
                        Exception(
                            f'validate_records@[{i}]: null value given for required property "{required_prop}"'
                        ),
                    )
                )
                continue

        for key, value in record.items():
            if key not in property_map:
                errors.append(
                    (
                        key,
                        Exception(
                            f'validate_records@[{i}]: unsupported property "{key}"'
                        ),
                    )
                )
                continue

            if not property_map[key].required and value is None:
                continue

            try:
                property_map[key].data_type.validate(value)
            except ValueError as e:
                errors.append((key, Exception(f"validate_records@[{i}]: {str(e)}")))

    return Result(errors=errors)
