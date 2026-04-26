import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    ErrorType, GraderComponentScore, GraderResult, InjectedError,
    InspectorAction, WorkerResponse,
)
from .tickets import TicketRecord


PRIORITY_LEVELS = ["low", "medium", "high", "urgent"]

CATEGORY_ALIASES = {
    "billing": ["billing", "payment", "invoice", "refund", "charge"],
    "technical": ["technical", "bug", "crash", "api", "integration", "webhook", "sso"],
    "account": ["account", "password", "login", "access", "delete", "lock", "security"],
    "product": ["product", "feature", "plan", "integration_inquiry", "comparison", "api_inquiry"],
    "shipping": ["shipping", "delivery", "order", "return", "package", "customs"],
}

DEPARTMENT_ALIASES = {
    "billing": ["billing", "billing_support", "finance"],
    "technical": ["technical", "technical_support", "engineering"],
    "account": ["account", "account_management"],
    "product": ["product", "product_team", "general_support"],
    "shipping": ["shipping", "logistics", "delivery"],
}

FIELD_ERRORS = {
    "classification": [ErrorType.CLASSIFICATION_WRONG],
    "priority": [ErrorType.PRIORITY_OFF],
    "department": [ErrorType.DEPARTMENT_WRONG],
    "response": [
        ErrorType.RESPONSE_MISSING_APOLOGY,
        ErrorType.RESPONSE_TOO_VAGUE,
        ErrorType.RESPONSE_NO_PERSONALIZATION,
    ],
    "resolution_actions": [ErrorType.RESOLUTION_INCOMPLETE, ErrorType.ESCALATION_MISSING],
}


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip(".")


def _field_matches_inspector_flag(field: str, inspector_field: str) -> bool:
    norm_f = _normalize(field)
    norm_i = _normalize(inspector_field)

    if norm_f == norm_i:
        return True

    if norm_f in norm_i or norm_i in norm_f:
        return True

    all_aliases = list(CATEGORY_ALIASES.keys()) + list(DEPARTMENT_ALIASES.keys())
    for alias_group in [CATEGORY_ALIASES, DEPARTMENT_ALIASES]:
        for cat, aliases in alias_group.items():
            if norm_f == cat and norm_i in aliases:
                return True
            if norm_i == cat and norm_f in aliases:
                return True

    return False


def grade_error_detection(
    injected_errors: List[InjectedError],
    inspector: InspectorAction,
) -> GraderComponentScore:
    if not injected_errors and not inspector.flagged:
        return GraderComponentScore(
            name="error_detection",
            score=1.0,
            max_weight=0.35,
            details="No errors injected and inspector correctly did not flag",
        )

    error_fields = set()
    for e in injected_errors:
        error_fields.add(e.field)

    if not error_fields:
        if inspector.flagged:
            return GraderComponentScore(
                name="error_detection",
                score=0.0,
                max_weight=0.35,
                details="No errors injected but inspector flagged something",
            )
        return GraderComponentScore(
            name="error_detection",
            score=1.0,
            max_weight=0.35,
            details="No errors injected and inspector correctly did not flag",
        )

    if not inspector.flagged and not inspector.flagged_fields:
        return GraderComponentScore(
            name="error_detection",
            score=0.0,
            max_weight=0.35,
            details=f"Missed all {len(error_fields)} error(s): {error_fields}",
        )

    detected = set()
    for flagged_field in inspector.flagged_fields:
        for error_field in error_fields:
            if _field_matches_inspector_flag(error_field, flagged_field):
                detected.add(error_field)
                break

    recall = len(detected) / len(error_fields) if error_fields else 1.0
    missed = error_fields - detected

    details_parts = [f"Detected: {len(detected)}/{len(error_fields)} errors"]
    if missed:
        details_parts.append(f"Missed: {missed}")

    return GraderComponentScore(
        name="error_detection",
        score=round(recall, 4),
        max_weight=0.35,
        details="; ".join(details_parts),
    )


def grade_precision(
    injected_errors: List[InjectedError],
    inspector: InspectorAction,
) -> GraderComponentScore:
    if not inspector.flagged and not inspector.flagged_fields:
        if injected_errors:
            return GraderComponentScore(
                name="precision",
                score=0.0,
                max_weight=0.25,
                details="No fields flagged despite errors existing",
            )
        return GraderComponentScore(
            name="precision",
            score=1.0,
            max_weight=0.25,
            details="No fields flagged, no errors present — correct",
        )

    error_fields = set()
    for e in injected_errors:
        error_fields.add(e.field)

    false_positives = 0
    true_positives = 0
    for flagged_field in inspector.flagged_fields:
        is_real_error = False
        for error_field in error_fields:
            if _field_matches_inspector_flag(error_field, flagged_field):
                is_real_error = True
                break
        if is_real_error:
            true_positives += 1
        else:
            false_positives += 1

    total_flags = true_positives + false_positives
    precision = true_positives / total_flags if total_flags > 0 else 1.0

    if false_positives > 0:
        false_positive_fields = []
        for flagged_field in inspector.flagged_fields:
            is_real = any(
                _field_matches_inspector_flag(ef, flagged_field) for ef in error_fields
            )
            if not is_real:
                false_positive_fields.append(flagged_field)
        details = f"False positives: {false_positive_fields} (flagged correct fields)"
    else:
        details = f"All {total_flags} flags were correct"

    return GraderComponentScore(
        name="precision",
        score=round(precision, 4),
        max_weight=0.25,
        details=details,
    )


def grade_issue_specificity(
    injected_errors: List[InjectedError],
    inspector: InspectorAction,
    ticket: TicketRecord,
) -> GraderComponentScore:
    if not injected_errors:
        if not inspector.issues:
            return GraderComponentScore(
                name="issue_specificity",
                score=1.0,
                max_weight=0.15,
                details="No issues to describe and none provided",
            )
        return GraderComponentScore(
            name="issue_specificity",
            score=0.0,
            max_weight=0.15,
            details="Issues described but no errors exist (false positives)",
        )

    if not inspector.issues:
        return GraderComponentScore(
            name="issue_specificity",
            score=0.0,
            max_weight=0.15,
            details=f"No issue descriptions provided despite {len(injected_errors)} error(s)",
        )

    specificity_scores = []
    error_fields = {e.field: e for e in injected_errors}

    for issue in inspector.issues:
        issue_field = _normalize(issue.get("field", ""))
        issue_reason = issue.get("reason", "").lower()

        matched_error = None
        for ef, error in error_fields.items():
            if _field_matches_inspector_flag(ef, issue_field):
                matched_error = error
                break

        if matched_error is None:
            specificity_scores.append(0.0)
            continue

        score = 0.3

        gt_keywords = {
            "classification": ["wrong", "incorrect", "should be", "correct category", "misclassified"],
            "priority": ["priority", "level", "urgent", "high", "medium", "low", "severity"],
            "department": ["department", "team", "routed", "wrong team", "correct department"],
            "response": ["response", "apology", "apologize", "vague", "personalize", "name", "generic"],
            "resolution_actions": ["resolution", "action", "missing", "incomplete", "escalat"],
        }
        relevant_kw = gt_keywords.get(matched_error.field, [])
        kw_hits = sum(1 for kw in relevant_kw if kw in issue_reason)
        if kw_hits > 0:
            score += min(kw_hits * 0.2, 0.4)

        error_specific_keywords = _extract_error_keywords(matched_error, ticket)
        if error_specific_keywords:
            specific_hits = sum(1 for kw in error_specific_keywords if kw in issue_reason)
            if specific_hits > 0:
                score += min(specific_hits * 0.15, 0.3)

        specificity_scores.append(min(score, 1.0))

    avg_specificity = sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0.0
    details = f"Average specificity: {avg_specificity:.2f} across {len(inspector.issues)} issue(s)"

    return GraderComponentScore(
        name="issue_specificity",
        score=round(avg_specificity, 4),
        max_weight=0.15,
        details=details,
    )


def _extract_error_keywords(error: InjectedError, ticket: TicketRecord) -> List[str]:
    keywords = []
    gt = ticket.ground_truth
    keywords.append(_normalize(gt.category))
    keywords.append(_normalize(gt.priority))
    if error.ground_truth_value and isinstance(error.ground_truth_value, str):
        keywords.append(_normalize(error.ground_truth_value))
    if error.corrupted_value and isinstance(error.corrupted_value, str):
        keywords.append(_normalize(error.corrupted_value))
    for term in gt.key_terms[:3]:
        keywords.append(term.lower())
    return keywords


def grade_correction_quality(
    injected_errors: List[InjectedError],
    inspector: InspectorAction,
) -> GraderComponentScore:
    if not injected_errors:
        if not inspector.suggested_corrections:
            return GraderComponentScore(
                name="correction_quality",
                score=1.0,
                max_weight=0.15,
                details="No corrections needed and none suggested",
            )
        return GraderComponentScore(
            name="correction_quality",
            score=0.5,
            max_weight=0.15,
            details="Corrections suggested but no errors exist",
        )

    if not inspector.suggested_corrections:
        return GraderComponentScore(
            name="correction_quality",
            score=0.0,
            max_weight=0.15,
            details=f"No corrections suggested despite {len(injected_errors)} error(s)",
        )

    error_fields = {e.field: e for e in injected_errors}
    correction_scores = []

    for cor_field, cor_value in inspector.suggested_corrections.items():
        matched_error = None
        for ef, error in error_fields.items():
            if _field_matches_inspector_flag(ef, cor_field):
                matched_error = error
                break

        if matched_error is None:
            correction_scores.append(0.0)
            continue

        gt_val = matched_error.ground_truth_value
        if isinstance(gt_val, str):
            gt_norm = _normalize(gt_val)
            cor_norm = _normalize(str(cor_value))
            if gt_norm == cor_norm:
                correction_scores.append(1.0)
            elif gt_norm in cor_norm or cor_norm in gt_norm:
                correction_scores.append(0.7)
            else:
                alias_groups = [CATEGORY_ALIASES, DEPARTMENT_ALIASES, PRIORITY_LEVELS]
                found_alias = False
                for group in alias_groups:
                    if isinstance(group, dict):
                        for cat, aliases in group.items():
                            gt_in = gt_norm in aliases or gt_norm == cat
                            cor_in = cor_norm in aliases or cor_norm == cat
                            if gt_in and cor_in:
                                correction_scores.append(0.5)
                                found_alias = True
                                break
                    elif isinstance(group, list):
                        if gt_norm in group and cor_norm in group:
                            correction_scores.append(0.5)
                            found_alias = True
                            break
                    if found_alias:
                        break
                if not found_alias:
                    correction_scores.append(0.2)
        elif isinstance(gt_val, list) and isinstance(cor_value, list):
            if set(str(x).lower() for x in gt_val) & set(str(x).lower() for x in cor_value):
                correction_scores.append(0.6)
            else:
                correction_scores.append(0.1)
        else:
            correction_scores.append(0.3)

    avg_correction = sum(correction_scores) / len(correction_scores) if correction_scores else 0.0
    details = f"Average correction quality: {avg_correction:.2f} across {len(inspector.suggested_corrections)} correction(s)"

    return GraderComponentScore(
        name="correction_quality",
        score=round(avg_correction, 4),
        max_weight=0.15,
        details=details,
    )


def grade_calibration(
    injected_errors: List[InjectedError],
    inspector: InspectorAction,
) -> GraderComponentScore:
    confidence = inspector.confidence

    if not injected_errors:
        if not inspector.flagged:
            if confidence <= 0.3:
                return GraderComponentScore(
                    name="calibration",
                    score=1.0,
                    max_weight=0.10,
                    details=f"Correct no-flag with low confidence ({confidence:.2f})",
                )
            else:
                score = max(0.0, 1.0 - (confidence - 0.3) * 2)
                return GraderComponentScore(
                    name="calibration",
                    score=round(score, 4),
                    max_weight=0.10,
                    details=f"Correct no-flag but overconfident ({confidence:.2f})",
                )
        else:
            penalty = confidence * 0.5
            return GraderComponentScore(
                name="calibration",
                score=round(max(0.0, 0.5 - penalty), 4),
                max_weight=0.10,
                details=f"False flag with confidence {confidence:.2f}",
            )

    error_fields = set(e.field for e in injected_errors)
    detected_count = 0
    for ff in inspector.flagged_fields:
        for ef in error_fields:
            if _field_matches_inspector_flag(ef, ff):
                detected_count += 1
                break

    recall = detected_count / len(error_fields) if error_fields else 0.0
    fp_count = max(0, len(inspector.flagged_fields) - detected_count)

    if recall >= 0.8 and fp_count == 0:
        if confidence >= 0.7:
            score = 1.0
        elif confidence >= 0.5:
            score = 0.8
        else:
            score = 0.6
    elif recall >= 0.5:
        if confidence <= 0.8:
            score = 0.7
        elif confidence > 0.9 and fp_count > 0:
            score = 0.3
        else:
            score = 0.5
    else:
        if confidence < 0.3:
            score = 0.6
        elif confidence > 0.7:
            score = 0.1
        else:
            score = 0.4

    details = (
        f"Confidence={confidence:.2f}, "
        f"Recall={recall:.2f}, "
        f"False positives={fp_count}"
    )

    return GraderComponentScore(
        name="calibration",
        score=round(score, 4),
        max_weight=0.10,
        details=details,
    )


def _is_flag_everything_hack(
    injected_errors: List[InjectedError], inspector: InspectorAction,
) -> bool:
    if not inspector.flagged:
        return False
    if len(inspector.flagged_fields) < 4:
        return False
    error_fields = {e.field for e in injected_errors}
    if len(error_fields) <= 1 and len(inspector.flagged_fields) >= 4:
        return True
    return False


def _is_max_confidence_hack(inspector: InspectorAction) -> bool:
    if inspector.confidence >= 0.99 and not inspector.flagged:
        return True
    if inspector.confidence >= 0.99 and not inspector.issues:
        return True
    return False


def _is_copy_paste_hack(inspector: InspectorAction) -> bool:
    if not inspector.issues or len(inspector.issues) < 2:
        return False
    reasons = [i.get("reason", "").lower().strip() for i in inspector.issues]
    if len(set(reasons)) == 1 and len(reasons) > 1:
        return True
    return False


def grade_inspector(
    injected_errors: List[InjectedError],
    inspector: InspectorAction,
    ticket: TicketRecord,
    task_type: str = "inspection_easy",
) -> GraderResult:
    scores = [
        grade_error_detection(injected_errors, inspector),
        grade_precision(injected_errors, inspector),
        grade_issue_specificity(injected_errors, inspector, ticket),
        grade_correction_quality(injected_errors, inspector),
        grade_calibration(injected_errors, inspector),
    ]

    weights = [0.35, 0.25, 0.15, 0.15, 0.10]
    total = sum(s.score * w for s, w in zip(scores, weights))

    if _is_flag_everything_hack(injected_errors, inspector):
        total *= 0.4

    if _is_max_confidence_hack(inspector):
        total *= 0.8

    if _is_copy_paste_hack(inspector):
        total *= 0.5

    overall = max(0.0, min(1.0, total))

    return GraderResult(
        component_scores=scores,
        overall_score=round(overall, 4),
        task_type=task_type,
    )
