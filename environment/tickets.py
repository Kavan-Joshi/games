import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TicketGroundTruth:
    category: str
    priority: str
    department: str
    difficulty: int = 1
    key_terms: List[str] = field(default_factory=list)
    required_response_elements: List[str] = field(default_factory=list)
    expected_resolution_actions: List[str] = field(default_factory=list)
    escalation_required: bool = False
    sentiment_should_address: bool = False


@dataclass
class TicketRecord:
    id: str
    subject: str
    body: str
    customer_id: str
    customer_name: str
    customer_tier: str
    customer_tenure_days: int
    sentiment: str
    previous_ticket_count: int
    ground_truth: TicketGroundTruth


@dataclass
class CustomerHistoryRecord:
    customer_id: str
    total_tickets: int
    resolved_tickets: int
    avg_satisfaction_score: float
    recent_issues: List[str]
    lifetime_value_usd: float
    last_contact_days_ago: int
    escalation_history: List[str]


TICKETS: List[TicketRecord] = [
    TicketRecord(
        id="TKT-001", subject="Double charge on my credit card",
        body="Hi, I noticed I was charged $49.99 twice on my credit card for the December subscription. The charges appeared on December 1st and December 3rd. I only have one account and I certainly didn't sign up twice. Can you please look into this and refund the duplicate charge? My credit card ending in 4521 was affected.",
        customer_id="CUS-101", customer_name="Sarah Mitchell", customer_tier="gold",
        customer_tenure_days=540, sentiment="negative", previous_ticket_count=3,
        ground_truth=TicketGroundTruth(
            category="billing", priority="high", department="billing_support",
            difficulty=1, key_terms=["charge", "duplicate", "refund", "credit card", "subscription"],
            required_response_elements=["acknowledge", "apologize", "investigate", "refund"],
            expected_resolution_actions=["refund_duplicate_charge", "verify_billing_records"],
        ),
    ),
    TicketRecord(
        id="TKT-002", subject="App crashes when uploading PDF files",
        body="Every time I try to upload a PDF file larger than 5MB, the application crashes completely. I've tried on Chrome and Firefox, same result. I'm using version 3.2.1 on Windows 11. This is blocking my work since I need to upload client reports daily. I've attached the crash log from Chrome's developer console.",
        customer_id="CUS-202", customer_name="James Rodriguez", customer_tier="silver",
        customer_tenure_days=180, sentiment="very_negative", previous_ticket_count=1,
        ground_truth=TicketGroundTruth(
            category="technical", priority="urgent", department="engineering",
            difficulty=2, key_terms=["crash", "upload", "PDF", "bug", "version"],
            required_response_elements=["acknowledge", "apologize", "workaround", "timeline"],
            expected_resolution_actions=["escalate_to_engineering", "provide_workaround", "create_bug_report"],
            escalation_required=True,
        ),
    ),
    TicketRecord(
        id="TKT-003", subject="Can't reset my password",
        body="I've been trying to reset my password for the past hour but I never receive the reset email. I've checked my spam folder and tried multiple times. My email address is j.chen@example.com. I need to access my account urgently for a project deadline tomorrow.",
        customer_id="CUS-303", customer_name="Jennifer Chen", customer_tier="bronze",
        customer_tenure_days=30, sentiment="negative", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="account", priority="high", department="account_management",
            difficulty=1, key_terms=["password", "reset", "email", "access", "spam"],
            required_response_elements=["acknowledge", "troubleshoot", "verify_email", "alternative_reset"],
            expected_resolution_actions=["verify_email_provider", "manual_password_reset", "check_spam_settings"],
        ),
    ),
    TicketRecord(
        id="TKT-004", subject="What integrations do you support?",
        body="I'm evaluating your platform for our company and need to know what third-party integrations are available. Specifically, we use Salesforce, Slack, and Jira. Also, do you have a REST API we could use for custom integrations? We're currently on the Enterprise plan trial.",
        customer_id="CUS-404", customer_name="David Park", customer_tier="platinum",
        customer_tenure_days=15, sentiment="neutral", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="product", priority="medium", department="product_team",
            difficulty=1, key_terms=["integration", "Salesforce", "Slack", "Jira", "API", "Enterprise"],
            required_response_elements=["list_integrations", "confirm_API", "offer_demo", "trial_info"],
            expected_resolution_actions=["provide_integration_list", "offer_technical_demo", "share_API_documentation"],
        ),
    ),
    TicketRecord(
        id="TKT-005", subject="Order hasn't arrived after 3 weeks",
        body="I ordered the Pro Starter Kit on November 10th and it still hasn't arrived. The tracking number you gave me (TRK-88234) hasn't been updated since November 12th - it just says 'Label Created'. I've called the shipping company twice and they can't locate the package. I'd like a replacement sent immediately or a full refund.",
        customer_id="CUS-505", customer_name="Maria Gonzalez", customer_tier="silver",
        customer_tenure_days=365, sentiment="very_negative", previous_ticket_count=2,
        ground_truth=TicketGroundTruth(
            category="shipping", priority="urgent", department="logistics",
            difficulty=1, key_terms=["order", "delivery", "tracking", "replacement", "refund", "package"],
            required_response_elements=["acknowledge", "apologize", "investigate_tracking", "resolution_option"],
            expected_resolution_actions=["contact_carrier", "send_replacement", "process_refund"],
            escalation_required=True,
        ),
    ),
    TicketRecord(
        id="TKT-006", subject="Invoice amount doesn't match quoted price",
        body="We signed up for the Annual Business plan at $199/month as quoted by your sales team (quote reference QT-4421). However, our latest invoice shows $249/month. I've attached both the quote and the invoice. The plan features are the same - we haven't added anything extra. Please correct this.",
        customer_id="CUS-606", customer_name="Robert Taylor", customer_tier="gold",
        customer_tenure_days=120, sentiment="negative", previous_ticket_count=1,
        ground_truth=TicketGroundTruth(
            category="billing", priority="high", department="billing_support",
            difficulty=1, key_terms=["invoice", "quote", "price", "discrepancy", "correction"],
            required_response_elements=["acknowledge", "investigate", "reference_quote", "correction_timeline"],
            expected_resolution_actions=["verify_quote", "correct_billing", "issue_credit"],
        ),
    ),
    TicketRecord(
        id="TKT-007", subject="API returning 403 Forbidden after token refresh",
        body="Our production integration started returning 403 errors yesterday after the OAuth token auto-refreshed. We're using the v2 API with client credentials flow. The token endpoint returns a valid token but subsequent API calls fail with 403. This is affecting 200+ users on our platform. Client ID: CLI-78945.",
        customer_id="CUS-707", customer_name="Alex Kumar", customer_tier="platinum",
        customer_tenure_days=400, sentiment="very_negative", previous_ticket_count=4,
        ground_truth=TicketGroundTruth(
            category="technical", priority="urgent", department="engineering",
            difficulty=2, key_terms=["API", "403", "token", "OAuth", "production", "client_credentials"],
            required_response_elements=["acknowledge", "apologize", "urgency", "investigation_steps", "escalation"],
            expected_resolution_actions=["escalate_to_engineering", "check_oauth_scopes", "verify_token_revocation", "provide_temp_workaround"],
            escalation_required=True,
        ),
    ),
    TicketRecord(
        id="TKT-008", subject="Delete my account and all data",
        body="Under GDPR, I request the complete deletion of my account and all associated personal data. My username is emma_wilson. I understand this is irreversible. Please confirm once the deletion is complete and provide a list of all data types that were removed.",
        customer_id="CUS-808", customer_name="Emma Wilson", customer_tier="bronze",
        customer_tenure_days=90, sentiment="neutral", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="account", priority="medium", department="account_management",
            difficulty=1, key_terms=["delete", "account", "GDPR", "data", "personal_information"],
            required_response_elements=["acknowledge", "confirm_process", "timeline", "data_types"],
            expected_resolution_actions=["initiate_gdpr_deletion", "confirm_identity", "schedule_data_removal"],
        ),
    ),
    TicketRecord(
        id="TKT-009", subject="Is there a way to bulk export user data?",
        body="We need to export all our user data (about 15,000 records) for compliance reporting. The current UI only lets me export 500 records at a time. Is there a bulk export feature or API endpoint we can use? We need the data in CSV format with all custom fields included.",
        customer_id="CUS-909", customer_name="Lisa Nakamura", customer_tier="gold",
        customer_tenure_days=300, sentiment="neutral", previous_ticket_count=5,
        ground_truth=TicketGroundTruth(
            category="product", priority="medium", department="product_team",
            difficulty=2, key_terms=["export", "bulk", "data", "CSV", "compliance", "API"],
            required_response_elements=["acknowledge", "bulk_option", "timeline", "format_options"],
            expected_resolution_actions=["provide_bulk_export_method", "check_API_limitations", "schedule_data_export"],
        ),
    ),
    TicketRecord(
        id="TKT-010", subject="Received damaged product - screen cracked",
        body="I just received my order #ORD-5567 and the tablet screen is cracked on arrival. The box was in good condition but the device itself has a visible crack across the display. I need a replacement as this was a gift for my daughter's birthday this weekend. I've taken photos of the damage.",
        customer_id="CUS-110", customer_name="Patricia Moore", customer_tier="silver",
        customer_tenure_days=60, sentiment="negative", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="shipping", priority="urgent", department="logistics",
            difficulty=1, key_terms=["damaged", "cracked", "replacement", "order", "photos"],
            required_response_elements=["acknowledge", "apologize", "request_photos", "replacement_timeline"],
            expected_resolution_actions=["request_damage_photos", "arrange_replacement", "provide_return_label"],
        ),
    ),
    TicketRecord(
        id="TKT-011", subject="Automatic renewal without notification",
        body="My account was auto-renewed for another year at $299 but I never received any renewal notification. I had planned to cancel before the renewal date as we've moved to a different platform. I'd like this charge reversed and my account downgraded to the free tier. Order #ORD-8834.",
        customer_id="CUS-211", customer_name="Thomas Anderson", customer_tier="silver",
        customer_tenure_days=420, sentiment="very_negative", previous_ticket_count=2,
        ground_truth=TicketGroundTruth(
            category="billing", priority="high", department="billing_support",
            difficulty=2, key_terms=["renewal", "notification", "cancel", "refund", "downgrade", "auto-renew"],
            required_response_elements=["acknowledge", "apologize", "review_policy", "refund_options"],
            expected_resolution_actions=["review_renewal_notification_logs", "process_refund", "cancel_subscription", "downgrade_account"],
            sentiment_should_address=True,
        ),
    ),
    TicketRecord(
        id="TKT-012", subject="SSO login not working with Okta",
        body="We configured SSO using SAML 2.0 with our Okta instance last week but users are getting 'Invalid SAML response' errors when trying to log in. Our IT team has verified the configuration matches your documentation. The issue seems intermittent - some users can log in fine while others get errors consistently. Entity ID: okta.prod.ourcompany.com",
        customer_id="CUS-312", customer_name="Michael Foster", customer_tier="gold",
        customer_tenure_days=250, sentiment="negative", previous_ticket_count=6,
        ground_truth=TicketGroundTruth(
            category="technical", priority="high", department="technical_support",
            difficulty=3, key_terms=["SSO", "SAML", "Okta", "login", "authentication", "Entity_ID"],
            required_response_elements=["acknowledge", "investigate", "SAML_troubleshooting", "escalation_plan"],
            expected_resolution_actions=["review_saml_configuration", "check_certificate", "verify_user_mapping", "coordinate_with_it_team"],
            escalation_required=True,
        ),
    ),
    TicketRecord(
        id="TKT-013", subject="Account locked after password attempts",
        body="My account got locked after I tried to log in 3 times with the wrong password (I was using an old password). I'm currently traveling for work and can't access any of my projects. My username is mk_therapist. Is there a way to unlock it quickly? I can verify my identity with my registered phone number.",
        customer_id="CUS-413", customer_name="Dr. Karen Liu", customer_tier="gold",
        customer_tenure_days=700, sentiment="very_negative", previous_ticket_count=1,
        ground_truth=TicketGroundTruth(
            category="account", priority="urgent", department="account_management",
            difficulty=1, key_terms=["locked", "password", "unlock", "verify", "access"],
            required_response_elements=["acknowledge", "urgency", "verify_identity", "unlock_timeline"],
            expected_resolution_actions=["verify_identity", "unlock_account", "reset_password"],
        ),
    ),
    TicketRecord(
        id="TKT-014", subject="Feature comparison between Professional and Enterprise",
        body="We're a 50-person company trying to decide between Professional ($49/user/mo) and Enterprise ($99/user/mo) plans. Main things we need: SSO, audit logs, API access, and custom workflows. Could you tell me exactly which features are in which plan? Also, is there volume pricing for 50+ seats?",
        customer_id="CUS-514", customer_name="Amanda Brooks", customer_tier="bronze",
        customer_tenure_days=5, sentiment="neutral", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="product", priority="low", department="product_team",
            difficulty=1, key_terms=["plan", "Professional", "Enterprise", "SSO", "audit", "API", "pricing"],
            required_response_elements=["feature_comparison", "pricing_info", "volume_discount", "recommendation"],
            expected_resolution_actions=["provide_plan_comparison", "quote_volume_pricing", "offer_trial", "connect_sales"],
        ),
    ),
    TicketRecord(
        id="TKT-015", subject="Wrong item shipped - received keyboard instead of mouse",
        body="I ordered a wireless ergonomic mouse (SKU: MSE-ERGO-001) but received a mechanical keyboard instead (the packing slip inside shows a completely different order number: ORD-7721). My actual order number is ORD-7734. I need the correct item shipped ASAP as I'm experiencing wrist pain and need the ergonomic mouse for work.",
        customer_id="CUS-615", customer_name="Daniel Kim", customer_tier="silver",
        customer_tenure_days=200, sentiment="negative", previous_ticket_count=1,
        ground_truth=TicketGroundTruth(
            category="shipping", priority="high", department="logistics",
            difficulty=1, key_terms=["wrong_item", "order", "SKU", "shipping_error", "replacement"],
            required_response_elements=["acknowledge", "apologize", "investigate_mixup", "shipment_timeline"],
            expected_resolution_actions=["return_wrong_item", "ship_correct_item", "investigate_warehouse_error"],
        ),
    ),
    TicketRecord(
        id="TKT-016", subject="Tax calculation error on invoice",
        body="Our invoice #INV-9932 has a tax calculation error. We're based in Texas (8.25% sales tax) but the invoice shows California tax rate (9.25%). Our billing address has been Texas since we signed up. This has been happening for the last 3 invoices. The overpayment amount is approximately $15/month. Please correct our tax jurisdiction and credit the overpayment.",
        customer_id="CUS-716", customer_name="Christopher Martinez", customer_tier="platinum",
        customer_tenure_days=600, sentiment="negative", previous_ticket_count=8,
        ground_truth=TicketGroundTruth(
            category="billing", priority="medium", department="billing_support",
            difficulty=2, key_terms=["tax", "invoice", "jurisdiction", "correction", "credit", "overpayment"],
            required_response_elements=["acknowledge", "investigate_tax_records", "correction_plan", "credit_timeline"],
            expected_resolution_actions=["correct_tax_jurisdiction", "issue_credit", "audit_historical_invoices", "prevent_recurrence"],
        ),
    ),
    TicketRecord(
        id="TKT-017", subject="Webhook notifications stopped working",
        body="All webhook notifications for our production environment stopped firing 2 days ago. We haven't changed any configuration. The webhook endpoint is healthy (we've tested it independently). Last successful delivery was on Dec 5th at 3:42 PM UTC. Webhook ID: WH-44521. This is critical for our order processing pipeline.",
        customer_id="CUS-817", customer_name="Rachel Green", customer_tier="gold",
        customer_tenure_days=350, sentiment="very_negative", previous_ticket_count=3,
        ground_truth=TicketGroundTruth(
            category="technical", priority="urgent", department="engineering",
            difficulty=2, key_terms=["webhook", "notification", "production", "delivery", "endpoint", "pipeline"],
            required_response_elements=["acknowledge", "urgency", "investigation", "status_update", "workaround"],
            expected_resolution_actions=["investigate_webhook_queue", "check_delivery_logs", "restart_webhook_worker", "provide_status_page"],
            escalation_required=True,
        ),
    ),
    TicketRecord(
        id="TKT-018", subject="Request to transfer account ownership",
        body="I'm leaving the company and need to transfer ownership of our team account to my colleague, Brian Scott (brian.scott@techcorp.io). Brian already has admin access on the account. The account ID is ACC-3349. Please let me know the process and how long it takes.",
        customer_id="CUS-918", customer_name="Samantha Lee", customer_tier="platinum",
        customer_tenure_days=800, sentiment="neutral", previous_ticket_count=2,
        ground_truth=TicketGroundTruth(
            category="account", priority="medium", department="account_management",
            difficulty=1, key_terms=["transfer", "ownership", "admin", "account", "colleague"],
            required_response_elements=["acknowledge", "process_steps", "verification", "timeline"],
            expected_resolution_actions=["verify_both_parties", "process_transfer", "update_billing_contact", "confirm_completion"],
        ),
    ),
    TicketRecord(
        id="TKT-019", subject="How to set up automated report scheduling?",
        body="I'd like to set up weekly automated reports to be sent to our management team every Monday at 8 AM. The reports should include: user engagement metrics, conversion rates, and revenue data. I can create the reports manually but can't find the scheduling feature. Is this available on our current Professional plan or do we need to upgrade?",
        customer_id="CUS-119", customer_name="Nicole Brown", customer_tier="silver",
        customer_tenure_days=150, sentiment="neutral", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="product", priority="low", department="general_support",
            difficulty=1, key_terms=["report", "scheduling", "automated", "weekly", "metrics", "Professional"],
            required_response_elements=["feature_availability", "setup_instructions", "plan_check", "alternative"],
            expected_resolution_actions=["guide_to_scheduling_feature", "check_plan_features", "provide_step_by_step"],
        ),
    ),
    TicketRecord(
        id="TKT-020", subject="Package returned to sender without delivery attempt",
        body="My package (ORD-9012) was returned to sender without any delivery attempt. The tracking shows 'Return to Sender - Address Incomplete' but my address is correct and has been verified. I've been at the same address for 10 years. This is the third time this has happened with your shipments. I need the package reshipped immediately with a confirmed delivery date.",
        customer_id="CUS-220", customer_name="William Harris", customer_tier="gold",
        customer_tenure_days=500, sentiment="very_negative", previous_ticket_count=5,
        ground_truth=TicketGroundTruth(
            category="shipping", priority="urgent", department="logistics",
            difficulty=2, key_terms=["returned", "address", "redelivery", "tracking", "confirmed_delivery"],
            required_response_elements=["acknowledge", "apologize", "investigate", "guaranteed_delivery"],
            expected_resolution_actions=["verify_address_in_system", "reship_with_signature", "escalate_recurring_issue", "apply_courier_upgrade"],
            escalation_required=True,
            sentiment_should_address=True,
        ),
    ),
    TicketRecord(
        id="TKT-021", subject="Need receipts for expense reimbursement",
        body="I need PDF receipts for all my payments from January to March 2024 for my company expense report. I can see the transactions in my billing history but there's no download option for individual receipts. Can you generate and send me all receipts as PDFs? My customer ID is CUS-321.",
        customer_id="CUS-321", customer_name="Steven Wright", customer_tier="bronze",
        customer_tenure_days=100, sentiment="neutral", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="billing", priority="low", department="billing_support",
            difficulty=1, key_terms=["receipt", "PDF", "expense", "billing", "download"],
            required_response_elements=["acknowledge", "confirm_request", "timeline", "format"],
            expected_resolution_actions=["generate_receipt_pdfs", "send_via_email", "enable_receipt_download"],
        ),
    ),
    TicketRecord(
        id="TKT-022", subject="Mobile app keeps logging me out every 5 minutes",
        body="The mobile app (version 4.1.2 on iOS 17.2) keeps logging me out every 5 minutes. I have to re-enter my credentials each time. I've already tried: clearing cache, reinstalling the app, and resetting my password. The issue started after the last app update. Other team members on Android are not experiencing this.",
        customer_id="CUS-422", customer_name="Jessica Turner", customer_tier="silver",
        customer_tenure_days=270, sentiment="negative", previous_ticket_count=2,
        ground_truth=TicketGroundTruth(
            category="technical", priority="high", department="technical_support",
            difficulty=2, key_terms=["mobile", "iOS", "logout", "session", "update", "cache"],
            required_response_elements=["acknowledge", "troubleshooting", "platform_specific", "workaround"],
            expected_resolution_actions=["investigate_ios_session_bug", "check_token_expiry", "provide_workaround", "report_to_mobile_team"],
        ),
    ),
    TicketRecord(
        id="TKT-023", subject="Someone is using my account fraudulently",
        body="I just received an email confirmation that someone changed my account email to a different address (fraud123@gmail.com). I did NOT make this change. I'm concerned my account has been compromised. I had payment information stored on file. Please lock the account immediately and help me secure it. This needs urgent attention.",
        customer_id="CUS-523", customer_name="Andrew Phillips", customer_tier="gold",
        customer_tenure_days=450, sentiment="very_negative", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="account", priority="urgent", department="account_management",
            difficulty=2, key_terms=["fraud", "compromised", "security", "email_change", "lock", "payment"],
            required_response_elements=["acknowledge", "urgent_action", "security_steps", "investigation"],
            expected_resolution_actions=["lock_account_immediately", "verify_identity", "revert_email_change", "audit_account_activity", "notify_payment_team"],
            escalation_required=True,
        ),
    ),
    TicketRecord(
        id="TKT-024", subject="Custom field limits on Enterprise plan",
        body="We're on the Enterprise plan and have hit what appears to be a 200 custom field limit. Our use case requires approximately 350 custom fields across different record types. Is this a hard limit or can it be increased? We're willing to discuss additional costs. Also, are there performance implications of having many custom fields?",
        customer_id="CUS-624", customer_name="Kevin Zhao", customer_tier="platinum",
        customer_tenure_days=300, sentiment="neutral", previous_ticket_count=7,
        ground_truth=TicketGroundTruth(
            category="product", priority="medium", department="product_team",
            difficulty=2, key_terms=["custom_field", "limit", "Enterprise", "increase", "performance"],
            required_response_elements=["acknowledge", "current_limit", "increase_options", "performance_impact"],
            expected_resolution_actions=["investigate_limit_options", "check_database_performance", "propose_solution", "escalate_to_product"],
        ),
    ),
    TicketRecord(
        id="TKT-025", subject="Need customs documentation for international shipment",
        body="I placed an order for delivery to Germany (ORD-4491) but the package has been held at customs for 2 weeks due to missing documentation. DHL says they need a commercial invoice and EUR.1 certificate from the shipper. Can you provide these documents? The order value is $1,200 and contains electronics.",
        customer_id="CUS-725", customer_name="Henrik Johansson", customer_tier="gold",
        customer_tenure_days=180, sentiment="negative", previous_ticket_count=1,
        ground_truth=TicketGroundTruth(
            category="shipping", priority="high", department="logistics",
            difficulty=2, key_terms=["customs", "international", "Germany", "documentation", "commercial_invoice", "DHL"],
            required_response_elements=["acknowledge", "customs_process", "document_preparation", "timeline"],
            expected_resolution_actions=["generate_commercial_invoice", "prepare_eur1_certificate", "coordinate_with_carrier", "expedite_customs"],
        ),
    ),
    TicketRecord(
        id="TKT-026", subject="Discount code not applying at checkout",
        body="I received a promotional email with discount code SAVE20 for 20% off my next purchase, but when I enter it at checkout it says 'Code expired'. The email was sent just 3 days ago and has no expiration date mentioned. I'm trying to order the Pro Bundle ($299). Can you honor this discount?",
        customer_id="CUS-826", customer_name="Olivia Carter", customer_tier="bronze",
        customer_tenure_days=45, sentiment="negative", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="billing", priority="medium", department="billing_support",
            difficulty=1, key_terms=["discount", "promo_code", "checkout", "expired", "honor"],
            required_response_elements=["acknowledge", "investigate_code", "resolution", "alternative"],
            expected_resolution_actions=["verify_promotion", "apply_manual_discount", "check_code_validity"],
        ),
    ),
    TicketRecord(
        id="TKT-027", subject="Data loss after workspace merge",
        body="After merging two workspaces last week (as suggested by your support team, ticket TKT-8801), we've discovered that approximately 3 months of project history from the secondary workspace is missing. The merge was supposed to combine everything but only recent data was transferred. We have critical client deliverables in that lost data. This is extremely urgent.",
        customer_id="CUS-927", customer_name="Marcus Reid", customer_tier="platinum",
        customer_tenure_days=900, sentiment="very_negative", previous_ticket_count=12,
        ground_truth=TicketGroundTruth(
            category="technical", priority="urgent", department="engineering",
            difficulty=3, key_terms=["data_loss", "workspace", "merge", "history", "recovery", "critical"],
            required_response_elements=["acknowledge", "apologize", "urgency", "recovery_plan", "timeline"],
            expected_resolution_actions=["escalate_to_engineering", "initiate_data_recovery", "check_backup_snapshots", "investigate_merge_process", "provide_temp_workspace"],
            escalation_required=True,
            sentiment_should_address=True,
        ),
    ),
    TicketRecord(
        id="TKT-028", subject="Can I use the platform for HIPAA-compliant data?",
        body="Our healthcare practice is considering your platform for patient management. We need to ensure full HIPAA compliance including BAA (Business Associate Agreement), encrypted data at rest and in transit, and audit logging. Do you offer a HIPAA-compliant plan? We'd also need SOC 2 Type II documentation.",
        customer_id="CUS-128", customer_name="Dr. Sarah Palmer", customer_tier="bronze",
        customer_tenure_days=10, sentiment="neutral", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="product", priority="medium", department="product_team",
            difficulty=2, key_terms=["HIPAA", "BAA", "compliance", "healthcare", "SOC2", "encryption"],
            required_response_elements=["acknowledge", "compliance_info", "BAA_availability", "security_features"],
            expected_resolution_actions=["provide_compliance_documentation", "offer_BAA", "schedule_security_review", "connect_compliance_team"],
        ),
    ),
    TicketRecord(
        id="TKT-029", subject="Change delivery address for in-transit order",
        body="I just realized I'll be moving this weekend and my order #ORD-3356 is scheduled for delivery next Tuesday. Can I update the delivery address to my new address? The package hasn't shipped yet according to tracking. New address: 456 Oak Avenue, Apt 3B, Portland, OR 97201.",
        customer_id="CUS-229", customer_name="Jennifer Adams", customer_tier="silver",
        customer_tenure_days=130, sentiment="neutral", previous_ticket_count=0,
        ground_truth=TicketGroundTruth(
            category="shipping", priority="medium", department="logistics",
            difficulty=1, key_terms=["address_change", "delivery", "in_transit", "order", "update"],
            required_response_elements=["acknowledge", "check_shipment_status", "address_update_process", "confirmation"],
            expected_resolution_actions=["verify_shipment_status", "update_address", "confirm_new_delivery_date"],
        ),
    ),
    TicketRecord(
        id="TKT-030", subject="Requesting access to beta API features",
        body="We've been selected for your developer beta program (confirmation code BETA-2245) and would like early access to the new analytics API endpoints mentioned in the beta announcement. We've already signed the NDA. Our use case involves building a real-time dashboard for our executive team. When can we expect API credentials?",
        customer_id="CUS-330", customer_name="Ryan O'Brien", customer_tier="gold",
        customer_tenure_days=220, sentiment="neutral", previous_ticket_count=3,
        ground_truth=TicketGroundTruth(
            category="product", priority="low", department="product_team",
            difficulty=2, key_terms=["beta", "API", "analytics", "developer", "NDA", "credentials"],
            required_response_elements=["acknowledge", "verify_beta_status", "access_timeline", "documentation"],
            expected_resolution_actions=["verify_beta_enrollment", "provision_API_credentials", "share_beta_documentation", "assign_developer liaison"],
        ),
    ),
]

CUSTOMER_HISTORIES: Dict[str, CustomerHistoryRecord] = {
    "CUS-101": CustomerHistoryRecord(
        customer_id="CUS-101", total_tickets=3, resolved_tickets=2,
        avg_satisfaction_score=3.5, recent_issues=["billing dispute", "plan upgrade"],
        lifetime_value_usd=599.88, last_contact_days_ago=45,
        escalation_history=[],
    ),
    "CUS-202": CustomerHistoryRecord(
        customer_id="CUS-202", total_tickets=1, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=99.00, last_contact_days_ago=0,
        escalation_history=[],
    ),
    "CUS-303": CustomerHistoryRecord(
        customer_id="CUS-303", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=0.00, last_contact_days_ago=30,
        escalation_history=[],
    ),
    "CUS-404": CustomerHistoryRecord(
        customer_id="CUS-404", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=0.00, last_contact_days_ago=15,
        escalation_history=[],
    ),
    "CUS-505": CustomerHistoryRecord(
        customer_id="CUS-505", total_tickets=2, resolved_tickets=1,
        avg_satisfaction_score=2.0, recent_issues=["late delivery"],
        lifetime_value_usd=249.00, last_contact_days_ago=21,
        escalation_history=["delivery_escalation"],
    ),
    "CUS-606": CustomerHistoryRecord(
        customer_id="CUS-606", total_tickets=1, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=["pricing concern"],
        lifetime_value_usd=2388.00, last_contact_days_ago=14,
        escalation_history=[],
    ),
    "CUS-707": CustomerHistoryRecord(
        customer_id="CUS-707", total_tickets=4, resolved_tickets=3,
        avg_satisfaction_score=3.0, recent_issues=["API rate limits", "webhook delays", "OAuth issues"],
        lifetime_value_usd=11988.00, last_contact_days_ago=7,
        escalation_history=["engineering_escalation"],
    ),
    "CUS-808": CustomerHistoryRecord(
        customer_id="CUS-808", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=29.00, last_contact_days_ago=90,
        escalation_history=[],
    ),
    "CUS-909": CustomerHistoryRecord(
        customer_id="CUS-909", total_tickets=5, resolved_tickets=4,
        avg_satisfaction_score=4.0, recent_issues=["data export", "custom fields", "report generation"],
        lifetime_value_usd=3564.00, last_contact_days_ago=3,
        escalation_history=[],
    ),
    "CUS-110": CustomerHistoryRecord(
        customer_id="CUS-110", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=79.99, last_contact_days_ago=60,
        escalation_history=[],
    ),
    "CUS-211": CustomerHistoryRecord(
        customer_id="CUS-211", total_tickets=2, resolved_tickets=1,
        avg_satisfaction_score=2.5, recent_issues=["cancel_request"],
        lifetime_value_usd=597.00, last_contact_days_ago=0,
        escalation_history=[],
    ),
    "CUS-312": CustomerHistoryRecord(
        customer_id="CUS-312", total_tickets=6, resolved_tickets=4,
        avg_satisfaction_score=3.0, recent_issues=["SSO setup", "user provisioning", "permissions"],
        lifetime_value_usd=23760.00, last_contact_days_ago=5,
        escalation_history=["engineering_escalation", "account_escalation"],
    ),
    "CUS-413": CustomerHistoryRecord(
        customer_id="CUS-413", total_tickets=1, resolved_tickets=1,
        avg_satisfaction_score=4.0, recent_issues=[],
        lifetime_value_usd=899.00, last_contact_days_ago=180,
        escalation_history=[],
    ),
    "CUS-514": CustomerHistoryRecord(
        customer_id="CUS-514", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=0.00, last_contact_days_ago=5,
        escalation_history=[],
    ),
    "CUS-615": CustomerHistoryRecord(
        customer_id="CUS-615", total_tickets=1, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=149.00, last_contact_days_ago=200,
        escalation_history=[],
    ),
    "CUS-716": CustomerHistoryRecord(
        customer_id="CUS-716", total_tickets=8, resolved_tickets=6,
        avg_satisfaction_score=3.5, recent_issues=["tax correction", "invoice review", "contract renewal"],
        lifetime_value_usd=17964.00, last_contact_days_ago=10,
        escalation_history=["billing_escalation"],
    ),
    "CUS-817": CustomerHistoryRecord(
        customer_id="CUS-817", total_tickets=3, resolved_tickets=2,
        avg_satisfaction_score=3.0, recent_issues=["webhook configuration", "API access"],
        lifetime_value_usd=4428.00, last_contact_days_ago=2,
        escalation_history=["engineering_escalation"],
    ),
    "CUS-918": CustomerHistoryRecord(
        customer_id="CUS-918", total_tickets=2, resolved_tickets=2,
        avg_satisfaction_score=5.0, recent_issues=["admin setup"],
        lifetime_value_usd=35640.00, last_contact_days_ago=60,
        escalation_history=[],
    ),
    "CUS-119": CustomerHistoryRecord(
        customer_id="CUS-119", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=588.00, last_contact_days_ago=150,
        escalation_history=[],
    ),
    "CUS-220": CustomerHistoryRecord(
        customer_id="CUS-220", total_tickets=5, resolved_tickets=3,
        avg_satisfaction_score=1.5, recent_issues=["address issues", "delivery problems", "wrong items"],
        lifetime_value_usd=1998.00, last_contact_days_ago=1,
        escalation_history=["logistics_escalation", "recurring_delivery_complaint"],
    ),
    "CUS-321": CustomerHistoryRecord(
        customer_id="CUS-321", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=87.00, last_contact_days_ago=100,
        escalation_history=[],
    ),
    "CUS-422": CustomerHistoryRecord(
        customer_id="CUS-422", total_tickets=2, resolved_tickets=1,
        avg_satisfaction_score=3.0, recent_issues=["app performance"],
        lifetime_value_usd=882.00, last_contact_days_ago=35,
        escalation_history=[],
    ),
    "CUS-523": CustomerHistoryRecord(
        customer_id="CUS-523", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=1497.00, last_contact_days_ago=450,
        escalation_history=[],
    ),
    "CUS-624": CustomerHistoryRecord(
        customer_id="CUS-624", total_tickets=7, resolved_tickets=5,
        avg_satisfaction_score=3.5, recent_issues=["feature requests", "custom fields", "API limitations"],
        lifetime_value_usd=29700.00, last_contact_days_ago=8,
        escalation_history=["product_escalation"],
    ),
    "CUS-725": CustomerHistoryRecord(
        customer_id="CUS-725", total_tickets=1, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=599.00, last_contact_days_ago=180,
        escalation_history=[],
    ),
    "CUS-826": CustomerHistoryRecord(
        customer_id="CUS-826", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=0.00, last_contact_days_ago=45,
        escalation_history=[],
    ),
    "CUS-927": CustomerHistoryRecord(
        customer_id="CUS-927", total_tickets=12, resolved_tickets=8,
        avg_satisfaction_score=2.0, recent_issues=["data migration", "workspace issues", "merge problems"],
        lifetime_value_usd=53820.00, last_contact_days_ago=1,
        escalation_history=["engineering_escalation", "management_escalation"],
    ),
    "CUS-128": CustomerHistoryRecord(
        customer_id="CUS-128", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=0.00, last_contact_days_ago=10,
        escalation_history=[],
    ),
    "CUS-229": CustomerHistoryRecord(
        customer_id="CUS-229", total_tickets=0, resolved_tickets=0,
        avg_satisfaction_score=0.0, recent_issues=[],
        lifetime_value_usd=299.00, last_contact_days_ago=130,
        escalation_history=[],
    ),
    "CUS-330": CustomerHistoryRecord(
        customer_id="CUS-330", total_tickets=3, resolved_tickets=2,
        avg_satisfaction_score=4.0, recent_issues=["API access", "beta features"],
        lifetime_value_usd=2658.00, last_contact_days_ago=20,
        escalation_history=[],
    ),
}


def get_ticket(ticket_id: str) -> Optional[TicketRecord]:
    for t in TICKETS:
        if t.id == ticket_id:
            return t
    return None


def get_tickets_by_difficulty(max_difficulty: int = 3) -> List[TicketRecord]:
    return [t for t in TICKETS if t.ground_truth.difficulty <= max_difficulty]


def get_customer_history(customer_id: str) -> Optional[CustomerHistoryRecord]:
    return CUSTOMER_HISTORIES.get(customer_id)
