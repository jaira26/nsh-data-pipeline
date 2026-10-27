"""Controlled vocabularies (data contract A5), in their canonical casing.

Two kinds of enum, checked in different places:
  * SOURCE_ENUMS - values copied from NSH's source systems. Drift there (new values,
    casing changes) is a warning from `anomalies.enum_violations`, not a schema failure.
  * DERIVED_ENUMS - values our own preprocessing computes. Anything else is a pipeline
    bug, so the processed schemas enforce them.
"""
from nsh_pipeline.preprocess.flatten import GROUP_SIZE_LABELS
from nsh_pipeline.preprocess.privacy import AGE_LABELS

CLIENT_STATUSES = ("Active", "Terminated", "Run Out", "Pending", "Implementation")
EMPLOYEE_STATUSES = ("Enrolled", "Not Enrolled", "COBRA Enrolled", "Other")
SWIPE_STATUSES = ("Approved", "Denied", "Pended", "Partial", "Hold")
SERVICE_CATEGORIES = ("Medical", "Drugs & Medicine", "Vision", "Dental")
CLAIM_STATUSES = (
    "Approved", "Approved - High Dollar", "New", "Denied", "Request More Info", "In Progress",
    "Wex Failed",
)
CLAIM_ESCALATIONS = ("Standard", "Potential Duplicate", "High Dollar", "Needs Review", "Problematic")
PRIORITIES = ("Critical", "High", "Medium", "Low")
TICKET_STATUSES = (
    "New", "In Progress", "On Hold", "Waiting on Carrier", "Waiting on Member",
    "Waiting on Nonstop", "Waiting on Other", "Waiting on Provider", "Closed",
)
TICKET_TYPES = ("Claim", "Customer Support", "Technical Support", "General Questions", "Member Advocacy")
TICKET_CATEGORIES = ("Substantiation",)
COVERAGE_TIERS = ("EE", "ES", "EC", "ECH", "EF", "FAM", "D")
BENEFIT_TYPES = ("Medical", "Dental", "Vision", "Rx", "Life")
CARRIERS = ("Aetna", "Cigna", "Anthem", "BSC", "BCBS")  # glossary "Carrier"; NSH to confirm the full list
LEDGER_SOURCES = ("wex", "landmark", "cvhs", "syhc", "altcare", "other", "manual")
GENDERS = ("M", "F")  # not in A5; the values the generator and NSH's census reports use

SOURCE_ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("clients", "status"): CLIENT_STATUSES,
    ("members", "gender"): GENDERS,
    ("members", "coverage_tier"): COVERAGE_TIERS,
    ("member_benefits", "benefit_type"): BENEFIT_TYPES,
    ("member_benefits", "carrier"): CARRIERS,
    ("member_benefits", "coverage_tier"): COVERAGE_TIERS,
    ("swipes", "status"): SWIPE_STATUSES,
    ("swipes", "service_category"): SERVICE_CATEGORIES,
    ("claims", "service_category"): SERVICE_CATEGORIES,
    ("claims", "status"): CLAIM_STATUSES,
    ("claims", "escalation"): CLAIM_ESCALATIONS,
    ("claims", "priority"): PRIORITIES,
    ("tickets", "type"): TICKET_TYPES,
    ("tickets", "category"): TICKET_CATEGORIES,
    ("tickets", "status"): TICKET_STATUSES,
    ("tickets", "priority"): PRIORITIES,
    ("bill_lines", "carrier"): CARRIERS,
    ("ledger", "source"): LEDGER_SOURCES,
}

# Raw fields whose casing preprocessing normalizes, so drift is only visible before cleaning.
RAW_ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("applications", "status"): CLIENT_STATUSES,
}

DERIVED_ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("clients", "group_size_bucket"): tuple(GROUP_SIZE_LABELS),
    ("members", "age_band"): tuple(AGE_LABELS),
    ("members", "employee_status"): EMPLOYEE_STATUSES,
}
