from datetime import datetime

import pytest

from server.errors import (
    NameEmptyError,
    NameTooLongError,
    NameValidationError,
    RelationshipValidationError,
)
from server.models import ModelProperty
from server.models import datatypes as dt
from server.models import validate as v


@pytest.fixture(scope="function")
def properties():
    now = datetime.now()
    props = [
        ModelProperty(
            name="id",
            display_name="ID",
            data_type=dt.String(),
            description="User ID",
            required=True,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="name",
            display_name="Name",
            data_type=dt.String(),
            description="Name",
            required=True,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="age",
            display_name="Age",
            data_type=dt.Long(),
            description="Age in years",
            required=True,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="height",
            display_name="Height",
            data_type=dt.Double(unit="inches"),
            description="Height in inches",
            required=True,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="sex",
            display_name="Sex",
            data_type=dt.Enumeration(items=dt.String(), enum=["M", "F"]),
            description="Sex",
            required=True,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="salutation",
            display_name="Salutation",
            data_type=dt.Array(
                items=dt.String(), enum=["Mr.", "Mrs.", "Ms.", "Dr.", "Esq."]
            ),
            description="Salutation",
            required=False,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="email",
            display_name="Email",
            data_type=dt.String(format=dt.StringSubtypeFormat.EMAIL),
            description="Email address",
            required=False,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="url",
            display_name="URL",
            data_type=dt.String(format=dt.StringSubtypeFormat.URL),
            description="URL",
            required=False,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="favorite_numbers",
            display_name="Favorite numbers",
            data_type=dt.Long(),
            description="Favorite numbers",
            required=False,
            created_at=now,
            updated_at=now,
        ),
        ModelProperty(
            name="favorite_color",
            display_name="Favorite color",
            data_type=dt.Enumeration(items=dt.String(), enum=["red", "green", "blue"]),
            description="Favorite color",
            required=False,
            created_at=now,
            updated_at=now,
        ),
    ]
    return props


def test_convert_properties(properties):
    schema = ModelProperty.schema()
    schema.dumps(properties[0])


def test_validate_model_and_model_property_names():
    v.validate_property_name("name")
    v.validate_property_name("name_with_underscores")
    v.validate_property_name("ñ")
    v.validate_property_name("nächste_straße_unicode")
    v.validate_property_name("nam3_w1th_number5")
    v.validate_property_name("_just_start1ng_with_und3rsc0re")
    v.validate_property_name("_just_start1ng_with_und3rsc0re")

    with pytest.raises(NameEmptyError):
        v.validate_property_name("")

    with pytest.raises(NameTooLongError):
        v.validate_property_name("x" * 100)

    with pytest.raises(NameEmptyError):
        v.validate_property_name("   \t \r   \n  ")

    with pytest.raises(NameValidationError):
        v.validate_property_name("@name")

    with pytest.raises(NameValidationError):
        v.validate_property_name("$name")

    with pytest.raises(NameValidationError):
        v.validate_property_name(".badname")

    with pytest.raises(NameValidationError):
        v.validate_property_name("3leading_number")

    with pytest.raises(NameValidationError):
        v.validate_property_name("-leading_number")

    with pytest.raises(NameValidationError):
        v.validate_property_name("foo-bar-baz")


def test_validate_relationship_name():
    v.validate_relationship_name("name")
    v.validate_relationship_name("name_with_underscores")
    v.validate_relationship_name("nam3_w1th_number5")
    v.validate_relationship_name("3leading_number")
    v.validate_relationship_name("name-with-dashes")  # To support UUIDs
    v.validate_relationship_name(
        "raw/wasExtractedFromAnatomicalRegion"
    )  # SPARC template
    v.validate_relationship_name("name.with.dots")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("   \t \r   \n  ")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("@name")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("$name")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("ŨnicodĘ")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("name with spaces")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("special*char")

    with pytest.raises(RelationshipValidationError):
        v.validate_relationship_name("_leading_underscore")


def test_valid_records(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Joseph Schmoe",
        "age": 40,
        "height": 72.0,
        "sex": "M",
    }
    result = v.validate_records(properties, records)
    assert result.ok


def test_invalid_record_id_value(properties):
    records = {
        "id": 1234,  # bad
        "name": "Joseph Schmoe",
        "height": 72.0,
        "age": 40,
        "sex": "M",
    }
    result = v.validate_records(properties, records)
    assert not result.ok


def test_missing_record_property(properties):
    # missing name, height, and age
    records = {"id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9", "sex": "M"}
    result = v.validate_records(properties, records)
    assert not result.ok
    assert "missing required property" in str(result.error("name")).lower()


def test_bad_value_for_type(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": 898989,
        "height": 72.0,
        "age": 40,
        "sex": "M",
    }
    result = v.validate_records(properties, records)
    assert "not a string: 898989" in str(result.error("name")).lower()


def test_bad_enumeration_value(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72.0,
        "age": 40,
        "sex": "G",
    }
    result = v.validate_records(properties, records)
    assert "[G] not in" in str(result.error("sex"))


def test_accept_integer_values_for_decimal_properties(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,  # specify height as a whole number
        "age": 40,
        "sex": "M",
    }
    result = v.validate_records(properties, records)
    assert result.ok


def test_accept_strings_with_valid_format_value(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,  # specify height as a whole number
        "age": 40,
        "sex": "M",
        "email": "schmoe.bro@hvac-dude.biz",
    }
    result = v.validate_records(properties, records)
    assert result.ok


def test_validate_array_data_types(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,  # specify height as a whole number
        "age": 40,
        "sex": "M",
        "email": "schmoe.bro@hvac-dude.biz",
        "favorite_numbers": ["red", "green"],
    }
    result = v.validate_records(properties, records)
    assert "Not a Long" in str(result.error("favorite_numbers"))


def test_reject_arrays_in_enum_data_types(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,
        "age": 40,
        "sex": "M",
        "favorite_color": ["red", "green"],
    }
    result = v.validate_records(properties, records)
    assert "not in enum ['red', 'green', 'blue']" in str(result.error("favorite_color"))


def test_reject_bad_enum_values(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,
        "age": 40,
        "sex": "M",
        "favorite_color": "purple",
    }
    result = v.validate_records(properties, records)
    assert "not in enum ['red', 'green', 'blue']" in str(result.error("favorite_color"))


def test_accept_enums_with_valid_values(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,
        "age": 40,
        "sex": "M",
        "favorite_color": "red",
    }
    result = v.validate_records(properties, records)
    assert result.ok


def test_accept_valid_email_uri(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,
        "age": 40,
        "sex": "M",
        "email": "schmoe.bro@hvac-dude.biz",
        "url": "http://www.hvac-dude.biz",
    }
    result = v.validate_records(properties, records)
    assert result.ok


def test_reject_invalid_email_uri(properties):
    records = {
        "id": "a009fee1-71f8-4b1f-aa62-7ad5ed91a4c9",
        "name": "Broseph Schmoe",
        "height": 72,
        "age": 40,
        "sex": "M",
        "email": "foobar",
        "url": "foobar",
    }
    result = v.validate_records(properties, records)
    assert "Not a String(Email): foobar" in str(result.error("email"))
    assert "Not a String(URL): foobar" in str(result.error("url"))
