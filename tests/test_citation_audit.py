from src.safety.citation_audit import citation_audit


def test_citation_audit_passes_with_available_citation():
    record = {"citations": ["R0001_C0000"], "risk_categories": ["absolute_claim"]}
    chunks = [{"chunk_id": "R0001_C0000", "source_id": "FTC"}]
    audit = citation_audit(record, chunks)
    assert audit["passes"] is True


def test_citation_audit_fails_with_missing_citation():
    record = {"citations": ["missing"], "risk_categories": ["absolute_claim"]}
    chunks = [{"chunk_id": "R0001_C0000", "source_id": "FTC"}]
    audit = citation_audit(record, chunks)
    assert audit["passes"] is False
