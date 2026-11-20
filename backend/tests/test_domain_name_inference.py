"""Brief analyzer domain name (P5b gap 5): a "for <determiner> ..." phrase names no domain."""

from app.services.domain_blueprint_service import _infer_domain_name


def test_determiner_phrases_are_not_a_domain():
    brief = "Given a user request and the list of tools available for that request, answer with the tool call."
    assert _infer_domain_name(brief) == "General Domain"
    assert _infer_domain_name("Rewrite each note, for example a short one, for wine importers.") == "Wine Importers"


def test_a_named_audience_still_names_the_domain():
    assert _infer_domain_name("A summarizer for the wine importers team.") == "The Wine Importers Team"
    assert _infer_domain_name("anything", "Explicit") == "Explicit"
