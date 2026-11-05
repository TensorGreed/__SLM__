"""HeldoutEvalRunRequest token budget: default unchanged, long structured answers allowed, still bounded."""
import pytest
from pydantic import ValidationError

from app.api.evaluation import HeldoutEvalRunRequest
from app.services.eval_task_handler_service import StructuredExtractionHandler


def test_default_max_new_tokens_unchanged():
    assert HeldoutEvalRunRequest(experiment_id=1).max_new_tokens == 128


def test_long_budget_accepted_and_reaches_multi_option_handler():
    req = HeldoutEvalRunRequest(experiment_id=1, max_new_tokens=6144)
    handler = StructuredExtractionHandler()
    handler._schema = {"scoring_mode": handler.SCORING_MODE_MULTI_OPTION}
    assert handler.max_new_tokens_override(req.max_new_tokens) == 6144


def test_plain_json_still_clamped_to_hardcap():
    handler = StructuredExtractionHandler()
    handler._schema = {}
    assert handler.max_new_tokens_override(8192) == handler.MAX_NEW_TOKENS_HARDCAP


@pytest.mark.parametrize("bad", [0, 8193])
def test_out_of_range_rejected(bad):
    with pytest.raises(ValidationError):
        HeldoutEvalRunRequest(experiment_id=1, max_new_tokens=bad)
