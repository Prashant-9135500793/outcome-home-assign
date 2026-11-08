"""Field-description hints: our rewrite of the field lists we were given.

The brief invites rewriting descriptions. We don't edit the incoming schema (keys must survive
and unknown schemas must work unchanged); instead we append hints, learned from the eval loop on
the sample set, to the descriptions of fields we recognise. Matching is by schema name and field
key; GENERIC hints apply to a field key in any schema (keys like `total_base_rent` mean the same
thing wherever they appear). A schema we have never seen simply gets no hints.
"""
from __future__ import annotations

from app.models import ExtractionSchema, SchemaField

_SHORT_PHRASE = ("Answer with a short phrase in the document's own words, without explanations or "
                 "parentheticals.")

GENERIC: dict[str, str] = {
    "total_base_rent": "Gross sum of scheduled base rent over the whole initial term: for each rent period, "
                       "monthly rent x number of months, summed. Do NOT subtract free rent or abatement.",
    "rent_commencement_date": "If base rent is abated / free for the first N months, rent commences on the first "
                              "day after that period. If there is no abatement, it equals the lease commencement date.",
    "free_rent_months": "Count only months of base rent abatement stated in the lease. 0 if none.",
    "lease_term_months": "Whole months from commencement to expiration inclusive (e.g. 2025-10-01 to 2027-04-30 is 19).",
}

BY_SCHEMA: dict[str, dict[str, str]] = {
    "lease_abstraction": {
        "property_name": "Prefer the building name on the cover/title page or where the lease names the "
                         "building. Never a street address.",
        "building_address": "Copy the address phrase as the lease body defines the premises' location (e.g. the "
                            "text after 'commonly known as' or 'located at'), including any building name that is "
                            "part of that phrase, with city, state and ZIP.",
        "base_rent_escalation": _SHORT_PHRASE + " Examples: '3% annually', 'fixed steps per schedule', 'none'.",
        "expense_structure": _SHORT_PHRASE,
        "electricity_provision": _SHORT_PHRASE,
        "right_to_sublet": _SHORT_PHRASE + " e.g. 'with landlord consent, not unreasonably withheld'.",
        "permitted_use": "The permitted use phrase only (e.g. 'general office purposes'), without restrictions "
                         "such as 'and no other purpose'.",
        "late_fee": "Quote the operative late-charge language of the lease (amount and any interest rate).",
        "holdover_rent": "Quote the operative holdover-rent language of the lease.",
    },
}


def apply_hints(schema: ExtractionSchema) -> ExtractionSchema:
    """Return a copy of the schema whose field descriptions carry our hints."""
    per_schema = BY_SCHEMA.get(schema.name, {})
    fields = {}
    for key, f in schema.fields.items():
        hint = per_schema.get(key) or GENERIC.get(key)
        desc = f"{f.description} Hint: {hint}" if hint else f.description
        fields[key] = SchemaField(name=f.name, description=desc, primary=f.primary)
    return ExtractionSchema(name=schema.name, description=schema.description, fields=fields)
