"""
Properties whose names start with this prefix should be discarded from any
models that are returned to the user.
"""
RESERVED_PREFIX = "@"


"""
Models cannot be named any of the given names, due to conflicts with
conventions used in the publishing process.
"""
RESERVED_MODEL_NAMES = set(["file"])


def is_reserved_model_name(name: str) -> bool:
    """
    Test if the property name is reserved.
    """
    return name in RESERVED_MODEL_NAMES


def is_reserved_property_name(name: str) -> bool:
    """
    Test if the property name is reserved.
    """
    return name.startswith(RESERVED_PREFIX)


def strip_reserved_prefix(name: str) -> str:
    """
    Removes `RESERVED_PREFIX` from `name`.
    """
    return name.lstrip(RESERVED_PREFIX)
