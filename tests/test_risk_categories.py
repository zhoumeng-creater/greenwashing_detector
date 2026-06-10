from src.generation.output_schema import RISK_CATEGORIES


def test_expected_risk_categories_present():
    assert "vague_general_claim" in RISK_CATEGORIES
    assert "unsubstantiated_claim" in RISK_CATEGORIES
    assert "absolute_claim" in RISK_CATEGORIES
    assert "carbon_offset_or_net_zero_risk" in RISK_CATEGORIES
