import random
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ErrorType, InjectedError, WorkerResponse,
)
from .tickets import TicketRecord


PRIORITY_LEVELS = ["low", "medium", "high", "urgent"]
CATEGORY_ALIASES = {
    "billing": ["payment", "invoice", "refund", "charge", "billing"],
    "technical": ["bug", "crash", "api", "integration", "webhook", "sso", "error"],
    "account": ["password", "login", "access", "delete", "lock", "security", "account"],
    "product": ["feature", "plan", "integration_inquiry", "comparison", "api_inquiry"],
    "shipping": ["shipping", "delivery", "order", "return", "package", "customs"],
}
DEPARTMENT_ALIASES = {
    "billing": ["billing_support", "finance"],
    "technical": ["technical_support", "engineering"],
    "account": ["account_management"],
    "product": ["product_team", "general_support"],
    "shipping": ["logistics", "delivery"],
}

GOOD_RESPONSES = {
    "negative": [
        "Dear {name}, I sincerely apologize for the inconvenience you've experienced. "
        "I understand how frustrating this must be, and I want to assure you that we are "
        "taking this matter seriously. I will investigate this right away and provide you "
        "with a resolution as soon as possible. You can expect an update within 24 hours.\n\n"
        "Best regards,\nCustomer Support Team",
        "Dear {name}, I'm truly sorry to hear about this issue. Your experience matters "
        "greatly to us, and I understand your frustration. Let me look into this immediately "
        "and work towards getting this resolved for you. I'll keep you informed every step "
        "of the way.\n\nBest regards,\nCustomer Support Team",
    ],
    "very_negative": [
        "Dear {name}, I deeply apologize for this unacceptable experience. I completely "
        "understand your frustration, and this is not the level of service we strive to "
        "provide. I am escalating this to our senior team immediately to ensure a swift "
        "resolution. You will hear back from us within the next few hours.\n\n"
        "Best regards,\nCustomer Support Team",
    ],
    "neutral": [
        "Dear {name}, thank you for reaching out to us. I'd be happy to help you with "
        "this. Let me look into your request and get back to you with the information "
        "you need. You can expect a response within 24 hours.\n\n"
        "Best regards,\nCustomer Support Team",
    ],
    "positive": [
        "Dear {name}, thank you for your positive feedback! We're glad to hear about "
        "your experience. If there's anything else we can help you with, please don't "
        "hesitate to reach out.\n\nBest regards,\nCustomer Support Team",
    ],
}

VAGUE_RESPONSES = [
    "We've received your request and will look into it.",
    "Thank you for contacting support. We will get back to you.",
    "Your ticket has been noted and someone will follow up.",
    "We appreciate you reaching out and will handle this accordingly.",
]

GENERIC_NO_NAME_RESPONSES = [
    "Dear customer, thank you for reaching out. We will look into your concern "
    "and get back to you shortly.\n\nBest regards,\nCustomer Support Team",
    "Hello, thank you for contacting us about this matter. We are reviewing "
    "your request and will follow up soon.\n\nBest regards,\nCustomer Support Team",
]


class ErrorInjector:
    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random(42)

    def generate_worker_response(
        self,
        ticket: TicketRecord,
        error_mix: Optional[Dict[str, float]] = None,
    ) -> Tuple[WorkerResponse, List[InjectedError]]:
        gt = ticket.ground_truth
        name = ticket.customer_name

        if error_mix is None:
            error_mix = {"clean": 0.3, "subtle": 0.4, "obvious": 0.2, "multi": 0.1}

        roll = self._rng.random()
        cumulative = 0.0
        error_severity = "clean"
        for key, prob in error_mix.items():
            cumulative += prob
            if roll < cumulative:
                error_severity = key
                break

        classification = gt.category
        priority = gt.priority
        department = gt.department
        resolution_actions = list(gt.expected_resolution_actions)
        response = self._build_good_response(ticket)
        errors: List[InjectedError] = []

        if error_severity == "clean":
            return WorkerResponse(
                classification=classification,
                priority=priority,
                department=department,
                response=response,
                resolution_actions=resolution_actions,
            ), errors

        if error_severity == "obvious":
            classification, errors = self._inject_classification_error(
                classification, gt, errors, subtlety=1
            )

        elif error_severity == "subtle":
            error_choice = self._rng.choice(["priority", "department", "response_apology", "response_vague"])
            if error_choice == "priority":
                priority, errors = self._inject_priority_error(priority, gt, errors, subtlety=2)
            elif error_choice == "department":
                department, errors = self._inject_department_error(department, gt, errors, subtlety=2)
            elif error_choice == "response_apology":
                response, errors = self._inject_missing_apology(response, ticket, errors, subtlety=2)
            elif error_choice == "response_vague":
                response, errors = self._inject_vague_response(response, ticket, errors, subtlety=2)

        elif error_severity == "multi":
            error_types_to_inject = self._rng.sample(
                ["classification", "priority", "response_apology", "response_no_name", "resolution"],
                k=2,
            )
            for et in error_types_to_inject:
                if et == "classification":
                    classification, errors = self._inject_classification_error(
                        classification, gt, errors, subtlety=2
                    )
                elif et == "priority":
                    priority, errors = self._inject_priority_error(priority, gt, errors, subtlety=1)
                elif et == "response_apology":
                    response, errors = self._inject_missing_apology(response, ticket, errors, subtlety=2)
                elif et == "response_no_name":
                    response, errors = self._inject_no_personalization(response, ticket, errors, subtlety=1)
                elif et == "resolution":
                    resolution_actions, errors = self._inject_incomplete_resolution(
                        resolution_actions, gt, errors, subtlety=1
                    )

        return WorkerResponse(
            classification=classification,
            priority=priority,
            department=department,
            response=response,
            resolution_actions=resolution_actions,
        ), errors

    def _build_good_response(self, ticket: TicketRecord) -> str:
        name = ticket.customer_name
        sentiment = ticket.sentiment
        templates = GOOD_RESPONSES.get(sentiment, GOOD_RESPONSES["neutral"])
        template = self._rng.choice(templates)
        return template.format(name=name)

    def _inject_classification_error(
        self, current: str, gt, errors: list, subtlety: int = 1,
    ) -> Tuple[str, list]:
        wrong_options = [c for c in CATEGORY_ALIASES if c != gt.category]
        if not wrong_options:
            return current, errors
        wrong = self._rng.choice(wrong_options)
        errors.append(InjectedError(
            error_type=ErrorType.CLASSIFICATION_WRONG,
            field="classification",
            ground_truth_value=gt.category,
            corrupted_value=wrong,
            subtlety=subtlety,
            description=f"Classification should be '{gt.category}' but got '{wrong}'",
        ))
        return wrong, errors

    def _inject_priority_error(
        self, current: str, gt, errors: list, subtlety: int = 1,
    ) -> Tuple[str, list]:
        idx = PRIORITY_LEVELS.index(gt.priority) if gt.priority in PRIORITY_LEVELS else 1
        if subtlety == 1:
            shift = self._rng.choice([-2, 2])
        else:
            shift = self._rng.choice([-1, 1])
        new_idx = max(0, min(len(PRIORITY_LEVELS) - 1, idx + shift))
        wrong = PRIORITY_LEVELS[new_idx]
        if wrong == gt.priority:
            wrong = PRIORITY_LEVELS[max(0, min(len(PRIORITY_LEVELS) - 1, idx + 1))]
        errors.append(InjectedError(
            error_type=ErrorType.PRIORITY_OFF,
            field="priority",
            ground_truth_value=gt.priority,
            corrupted_value=wrong,
            subtlety=subtlety,
            description=f"Priority should be '{gt.priority}' but got '{wrong}'",
        ))
        return wrong, errors

    def _inject_department_error(
        self, current: str, gt, errors: list, subtlety: int = 1,
    ) -> Tuple[str, list]:
        wrong_options = [d for d in DEPARTMENT_ALIASES if d != gt.department]
        if not wrong_options:
            return current, errors
        wrong_cat = self._rng.choice(wrong_options)
        wrong_dept = self._rng.choice(DEPARTMENT_ALIASES[wrong_cat])
        errors.append(InjectedError(
            error_type=ErrorType.DEPARTMENT_WRONG,
            field="department",
            ground_truth_value=gt.department,
            corrupted_value=wrong_dept,
            subtlety=subtlety,
            description=f"Department should be '{gt.department}' but got '{wrong_dept}'",
        ))
        return wrong_dept, errors

    def _inject_missing_apology(
        self, response: str, ticket: TicketRecord, errors: list, subtlety: int = 1,
    ) -> Tuple[str, list]:
        name = ticket.customer_name
        no_apology = (
            f"Dear {name}, thank you for reaching out regarding this matter. "
            f"I have reviewed your request and will be taking the necessary steps "
            f"to address it. You can expect a follow-up from us within 24 hours.\n\n"
            f"Best regards,\nCustomer Support Team"
        )
        errors.append(InjectedError(
            error_type=ErrorType.RESPONSE_MISSING_APOLOGY,
            field="response",
            ground_truth_value="[should contain apology]",
            corrupted_value="[apology missing]",
            subtlety=subtlety,
            description=f"Response is missing apology for {ticket.sentiment} sentiment customer",
        ))
        return no_apology, errors

    def _inject_vague_response(
        self, response: str, ticket: TicketRecord, errors: list, subtlety: int = 1,
    ) -> Tuple[str, list]:
        vague = self._rng.choice(VAGUE_RESPONSES)
        errors.append(InjectedError(
            error_type=ErrorType.RESPONSE_TOO_VAGUE,
            field="response",
            ground_truth_value="[should be detailed]",
            corrupted_value=vague,
            subtlety=subtlety,
            description="Response is too vague and lacks actionable content",
        ))
        return vague, errors

    def _inject_no_personalization(
        self, response: str, ticket: TicketRecord, errors: list, subtlety: int = 1,
    ) -> Tuple[str, list]:
        generic = self._rng.choice(GENERIC_NO_NAME_RESPONSES)
        errors.append(InjectedError(
            error_type=ErrorType.RESPONSE_NO_PERSONALIZATION,
            field="response",
            ground_truth_value=f"[should use name: {ticket.customer_name}]",
            corrupted_value="[no personalization]",
            subtlety=subtlety,
            description=f"Response does not personalize for customer '{ticket.customer_name}'",
        ))
        return generic, errors

    def _inject_incomplete_resolution(
        self, actions: list, gt, errors: list, subtlety: int = 1,
    ) -> Tuple[list, list]:
        if len(actions) <= 1:
            return actions, errors
        removed = self._rng.choice(actions)
        new_actions = [a for a in actions if a != removed]
        errors.append(InjectedError(
            error_type=ErrorType.RESOLUTION_INCOMPLETE,
            field="resolution_actions",
            ground_truth_value=actions,
            corrupted_value=new_actions,
            subtlety=subtlety,
            description=f"Missing resolution action: '{removed}'",
        ))
        return new_actions, errors
