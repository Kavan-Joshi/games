import pytest
from environment.models import (
    ErrorType, InjectedError, InspectorAction, TaskType,
)


def _make_injected_error(field="classification", error_type=ErrorType.CLASSIFICATION_WRONG,
                          ground_truth="billing", corrupted="technical", subtlety=1):
    return InjectedError(
        error_type=error_type,
        field=field,
        ground_truth_value=ground_truth,
        corrupted_value=corrupted,
        subtlety=subtlety,
    )


class TestGraderComponents:
    def test_error_detection_catches_real_error(self):
        from environment.graders import grade_error_detection
        errors = [_make_injected_error("classification")]
        inspector = InspectorAction(flagged=True, flagged_fields=["classification"], confidence=0.7)
        result = grade_error_detection(errors, inspector)
        assert result.score > 0.5

    def test_error_detection_misses_error(self):
        from environment.graders import grade_error_detection
        errors = [_make_injected_error("classification")]
        inspector = InspectorAction(flagged=False, confidence=0.2)
        result = grade_error_detection(errors, inspector)
        assert result.score < 0.3

    def test_precision_flags_clean_response(self):
        from environment.graders import grade_precision
        errors = []
        inspector = InspectorAction(flagged=True, flagged_fields=["priority"], confidence=0.8)
        result = grade_precision(errors, inspector)
        assert result.score < 0.5

    def test_precision_correctly_ignores_clean(self):
        from environment.graders import grade_precision
        errors = []
        inspector = InspectorAction(flagged=False, confidence=0.2)
        result = grade_precision(errors, inspector)
        assert result.score > 0.5

    def test_calibration_good_confidence(self):
        from environment.graders import grade_calibration
        errors = [_make_injected_error("priority")]
        inspector = InspectorAction(flagged=True, flagged_fields=["priority"], confidence=0.8)
        result = grade_calibration(errors, inspector)
        assert result.score > 0.3

    def test_calibration_overconfident(self):
        from environment.graders import grade_calibration
        errors = []
        inspector = InspectorAction(flagged=True, flagged_fields=["priority"], confidence=0.99)
        result = grade_calibration(errors, inspector)
        assert result.score < 0.5


class TestGraderAntiHacks:
    def test_flag_everything_hack(self):
        from environment.graders import _is_flag_everything_hack
        errors = []
        inspector = InspectorAction(
            flagged=True,
            flagged_fields=["classification", "priority", "department", "response"],
            confidence=0.8,
        )
        assert _is_flag_everything_hack(errors, inspector) is True

    def test_not_flag_everything(self):
        from environment.graders import _is_flag_everything_hack
        errors = [_make_injected_error("classification")]
        inspector = InspectorAction(flagged=True, flagged_fields=["classification"], confidence=0.7)
        assert _is_flag_everything_hack(errors, inspector) is False

    def test_max_confidence_hack(self):
        from environment.graders import _is_max_confidence_hack
        inspector = InspectorAction(flagged=True, confidence=0.99)
        assert _is_max_confidence_hack(inspector) is True

    def test_not_max_confidence(self):
        from environment.graders import _is_max_confidence_hack
        inspector = InspectorAction(flagged=True, confidence=0.8)
        assert _is_max_confidence_hack(inspector) is False

    def test_copy_paste_hack(self):
        from environment.graders import _is_copy_paste_hack
        inspector = InspectorAction(
            flagged=True,
            flagged_fields=["classification", "priority"],
            issues=[
                {"field": "classification", "reason": "Error detected"},
                {"field": "priority", "reason": "Error detected"},
            ],
            confidence=0.7,
        )
        assert _is_copy_paste_hack(inspector) is True

    def test_not_copy_paste(self):
        from environment.graders import _is_copy_paste_hack
        inspector = InspectorAction(
            flagged=True,
            flagged_fields=["classification", "priority"],
            issues=[
                {"field": "classification", "reason": "Wrong category for billing issue"},
                {"field": "priority", "reason": "Should be urgent for negative sentiment"},
            ],
            confidence=0.7,
        )
        assert _is_copy_paste_hack(inspector) is False


class TestGradeInspector:
    def test_excellent_inspector(self):
        from environment.graders import grade_inspector
        from environment.tickets import TICKETS
        ticket = TICKETS[0]
        errors = [_make_injected_error("classification")]
        inspector = InspectorAction(
            flagged=True,
            flagged_fields=["classification"],
            issues=[{"field": "classification", "reason": "Wrong category"}],
            suggested_corrections={"classification": "billing"},
            confidence=0.85,
        )
        result = grade_inspector(errors, inspector, ticket, "inspection_easy")
        assert result.overall_score > 0.5

    def test_terrible_inspector(self):
        from environment.graders import grade_inspector
        from environment.tickets import TICKETS
        ticket = TICKETS[0]
        errors = [_make_injected_error("classification")]
        inspector = InspectorAction(flagged=False, confidence=0.9)
        result = grade_inspector(errors, inspector, ticket, "inspection_easy")
        assert result.overall_score < 0.5

    def test_hack_penalty_reduces_score(self):
        from environment.graders import grade_inspector
        from environment.tickets import TICKETS
        ticket = TICKETS[0]
        errors = []
        honest = InspectorAction(flagged=False, confidence=0.2)
        hacker = InspectorAction(
            flagged=True,
            flagged_fields=["classification", "priority", "department", "response"],
            confidence=0.99,
        )
        honest_result = grade_inspector(errors, honest, ticket, "inspection_easy")
        hacker_result = grade_inspector(errors, hacker, ticket, "inspection_easy")
        assert honest_result.overall_score > hacker_result.overall_score

    def test_five_components_returned(self):
        from environment.graders import grade_inspector
        from environment.tickets import TICKETS
        ticket = TICKETS[0]
        inspector = InspectorAction(flagged=False, confidence=0.5)
        result = grade_inspector([], inspector, ticket, "inspection_easy")
        assert len(result.component_scores) == 5
        names = [cs.name for cs in result.component_scores]
        assert "error_detection" in names
        assert "precision" in names
