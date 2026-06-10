from src.generation.output_schema import fallback_output, validate_output


def test_fallback_output_matches_schema():
    record = fallback_output("100% eco-friendly", "test")
    assert validate_output(record) == []
