from server.models import RelationshipType, get_relationship_type


def test_get_relationship_type():
    assert get_relationship_type("belongs_to") == RelationshipType("BELONGS_TO")
    assert get_relationship_type(
        "belongs_to_478e215d-04ec-4cdf-ac8b-d5289601c9f7"
    ) == RelationshipType("BELONGS_TO")

    assert get_relationship_type(
        "raw/wasExtractedFromAnatomicalRegion"
    ) == RelationshipType("RAW_WASEXTRACTEDFROMANATOMICALREGION")

    assert get_relationship_type("name.with.dots") == RelationshipType("NAME_WITH_DOTS")
