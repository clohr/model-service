from datetime import datetime, timezone

import pytest

from server.models import datatypes as dt


def test_simple_type_decoding():
    assert dt.deserialize('"Boolean"') == dt.Boolean()
    assert dt.deserialize('"Double"') == dt.Double()
    assert dt.deserialize('"Long"') == dt.Long()
    assert dt.deserialize('"String"') == dt.String()

    assert dt.deserialize("Boolean") == dt.Boolean()
    assert dt.deserialize("Double") == dt.Double()
    assert dt.deserialize("Long") == dt.Long()
    assert dt.deserialize("String") == dt.String()


def test_deserialize_complex_type_decoding():
    assert dt.deserialize("""{ "type": "Boolean" }""") == dt.Boolean()
    assert dt.deserialize("""{ "type": "Double" }""") == dt.Double()
    assert dt.deserialize("""{ "type": "Long" }""") == dt.Long()
    assert dt.deserialize("""{ "type": "String" }""") == dt.String()


def test_deserialize_double_with_format():
    assert dt.deserialize("""{ "type": "Double", "unit": null }""") == dt.Double(
        unit=None
    )
    assert dt.deserialize("""{ "type": "Double", "unit": "ms" }""") == dt.Double(
        unit="ms"
    )
    assert dt.deserialize("""{ "type": "Double", "unit": "" }""") == dt.Double(
        unit=None
    )
    with pytest.raises(Exception):
        assert dt.deserialize("""{ "type": "Double", "unit": 99 }""")


def test_deserialize_long_with_format():
    assert dt.deserialize("""{ "type": "Long", "unit": null }""") == dt.Long(unit=None)
    assert dt.deserialize("""{ "type": "Long", "unit": "ms" }""") == dt.Long(unit="ms")
    assert dt.deserialize("""{ "type": "Long", "unit": "" }""") == dt.Long(unit=None)
    with pytest.raises(Exception):
        assert dt.deserialize("""{ "type": "Long", "unit": 99 }""")


def test_deserialize_string_with_format():
    assert dt.deserialize("""{ "type": "String", "format": null }""") == dt.String(
        format=None
    )
    # Not a string value:
    with pytest.raises(Exception):
        assert dt.deserialize("""{ "type": "String", "format": 99 }""")
    # Invalid format:
    with pytest.raises(Exception):
        assert dt.deserialize("""{ "type": "String", "format": "bar" }""")
    # Allowed formats:
    assert dt.deserialize("""{ "type": "String", "format": "email" }""") == dt.String(
        format=dt.StringSubtypeFormat.EMAIL
    )
    assert dt.deserialize("""{ "type": "String", "format": "date" }""") == dt.String(
        format=dt.StringDateFormat.DATE
    )
    assert dt.deserialize(
        """{ "type": "String", "format": "datetime" }"""
    ) == dt.String(format=dt.StringDateFormat.DATETIME)
    assert dt.deserialize(
        """{ "type": "String", "format": "date-time" }"""
    ) == dt.String(format=dt.StringDateFormat.DATETIME)
    assert dt.deserialize("""{ "type": "String", "format": "time" }""") == dt.String(
        format=dt.StringDateFormat.TIME
    )
    assert dt.deserialize("""{ "type": "String", "format": "url" }""") == dt.String(
        format=dt.StringSubtypeFormat.URL
    )
    assert dt.deserialize("""{ "type": "String", "format": "url" }""") == dt.String(
        format=dt.StringSubtypeFormat.URL
    )


def test_deserialize_array():
    assert dt.deserialize(
        """{ "type": "array", "items": { "type": "Boolean", "enum": [true] } }"""
    ) == dt.Array(items=dt.Boolean(), enum=[True])

    assert dt.deserialize(
        """{ "type": "array", "items": { "type": "Date" } }"""
    ) == dt.Array(items=dt.Date())

    assert dt.deserialize(
        """{ "type": "array", "items": { "type": "Boolean" } }"""
    ) == dt.Array(items=dt.Boolean())

    with pytest.raises(Exception):
        assert dt.deserialize(
            """{ "type": "array", "items": { "type": "Boolean", "enum": ["foo"] } }"""
        )


def test_deserialize_enum():
    assert dt.deserialize(
        """{ "type": "enum", "items": { "type": "Boolean", "enum": [true] } }"""
    ) == dt.Enumeration(items=dt.Boolean(), enum=[True])

    assert dt.deserialize(
        """{ "type": "enum", "items": { "type": "String", "enum": [] }}"""
    ) == dt.Enumeration(items=dt.String(), enum=[])

    with pytest.raises(Exception):
        assert dt.deserialize(
            """{ "type": "enum", "items": { "type": "Boolean", "enum": ["foo"] } }"""
        )
    with pytest.raises(Exception):
        assert dt.deserialize(
            """{ "type": "enum", "items": { "type": "Boolean" } }"""
        ) == dt.Enumeration(items=dt.Boolean())


def test_serialize_simple_types():
    assert dt.serialize(dt.Boolean()) == """{"type": "Boolean"}"""
    assert dt.serialize(dt.String()) == """{"type": "String", "format": null}"""
    assert (
        dt.serialize(dt.String(format=dt.StringSubtypeFormat.EMAIL))
        == """{"type": "String", "format": "Email"}"""
    )
    assert dt.serialize(dt.Double()) == """{"type": "Double", "unit": null}"""
    assert (
        dt.serialize(dt.Double(unit="inches"))
        == """{"type": "Double", "unit": "inches"}"""
    )
    assert dt.serialize(dt.Long()) == """{"type": "Long", "unit": null}"""
    assert (
        dt.serialize(dt.Long(unit="inches")) == """{"type": "Long", "unit": "inches"}"""
    )


def test_serialize_complex_types():
    assert (
        dt.serialize(dt.Array(items=dt.Boolean(), enum=[True]))
        == """{"type": "Array", "items": {"type": "Boolean", "enum": [true]}}"""
    )
    assert (
        dt.serialize(dt.Array(items=dt.Boolean()))
        == """{"type": "Array", "items": {"type": "Boolean"}}"""
    )
    assert (
        dt.serialize(
            dt.Array(
                items=dt.String(format=dt.StringSubtypeFormat.EMAIL),
                enum=["foo@bar.org", "xyz@abc.def-ghi.com"],
            )
        )
        == """{"type": "Array", "items": {"type": "String", "format": "Email", "enum": ["foo@bar.org", "xyz@abc.def-ghi.com"]}}"""
    )
    assert (
        dt.serialize(dt.Array(items=dt.Double(), enum=[1, 2, 3]))
        == """{"type": "Array", "items": {"type": "Double", "unit": null, "enum": [1, 2, 3]}}"""
    )
    assert (
        dt.serialize(dt.Array(items=dt.Double(unit="inches"), enum=[1, 2, 3]))
        == """{"type": "Array", "items": {"type": "Double", "unit": "inches", "enum": [1, 2, 3]}}"""
    )


def test_simple_representation():
    # Boolean
    SimpleBoolean = dt.Boolean()
    assert SimpleBoolean.is_simple
    assert SimpleBoolean.into_simple() == "Boolean"

    # Long
    SimpleLong = dt.Long()
    assert SimpleLong.is_simple
    assert SimpleLong.into_simple() == "Long"

    ComplexLong = dt.Long(unit="m/sec")
    assert not ComplexLong.is_simple

    # Double
    SimpleDouble = dt.Double()
    assert SimpleDouble.is_simple
    assert SimpleDouble.into_simple() == "Double"

    # String
    SimpleString = dt.String()
    assert SimpleString.is_simple
    assert SimpleString.into_simple() == "String"

    ComplexString = dt.String(format=dt.StringSubtypeFormat.EMAIL)
    assert not ComplexString.is_simple
    assert ComplexString.into_simple() is None

    # Date
    SimpleDate = dt.Date()
    assert SimpleDate.is_simple
    assert SimpleDate.into_simple() == "Date"

    Array_ = dt.Array(
        items=dt.String(format=dt.StringSubtypeFormat.EMAIL),
        enum=["foo@bar.org", "xyz@abc.def-ghi.com"],
    )
    assert not Array_.is_simple
    assert Array_.into_simple() is None

    Enum = dt.Enumeration(items=dt.Boolean(), enum=[True])
    assert not Enum.is_simple
    assert Enum.into_simple() is None


def test_coercion():
    assert dt.Date().into("2004-05-05 00:00:00") == datetime(
        year=2004, month=5, day=5, tzinfo=timezone.utc
    )
    assert dt.Date().into("2019-11-16T08:00:00.000+10:00") == datetime(
        year=2019, month=11, day=15, hour=22, tzinfo=timezone.utc
    )
    assert dt.Date().into(datetime(year=2004, month=5, day=5)) == datetime(
        year=2004, month=5, day=5, tzinfo=timezone.utc
    )
    assert dt.Array(items=dt.Date()).into(
        ["2004-05-05 00:00:00", "2019-11-16T08:00:00.000+10:00"]
    ) == [
        datetime(year=2004, month=5, day=5, tzinfo=timezone.utc),
        datetime(year=2019, month=11, day=15, hour=22, tzinfo=timezone.utc),
    ]

    assert dt.Long().into("2004") == 2004
    assert dt.Array(items=dt.Long()).into(["2004", "2005"]) == [2004, 2005]

    assert dt.Double().into("3.14159") == 3.14159
    assert dt.Array(items=dt.Double()).into(["3.14159", "3.12"]) == [3.14159, 3.12]

    assert dt.Boolean().into("true") is True
    assert dt.Array(items=dt.Boolean()).into(["true", "false"]) == [True, False]

    with pytest.raises(ValueError):
        dt.Array(items=dt.Long(), enum=[1, 2, 3]).into([1, 2, 4])
