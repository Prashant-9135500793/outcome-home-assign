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
        "property_name": "The building or project name as the lease titles it, usually printed under the "
                         "document title on the cover page (e.g. 'OFFICE BUILDING LEASE' / 'Harlow Office Building'), or "
                         "on a reference page ('BUILDING: Hollis Office Plaza'). Copy it as printed there, even when it "
                         "reads like an address (e.g. '250 South Harbor'); do not add 'Street', city or ZIP. Null only "
                         "if the lease never names the building.",
        "building_address": "When the lease body says the premises are 'commonly known as' or 'located at' an "
                            "address, copy everything after those words through the ZIP code, unchanged, including a "
                            "building name written there (e.g. 'ONE HARLOW PLACE, 850 N. HARLOW AVENUE, Naperville, "
                            "Illinois 60540'). Otherwise give the street address with city, state and ZIP. A building "
                            "name on its own labelled line (e.g. 'BUILDING: X' above the street) is the property name, "
                            "not part of the address.",
        "base_rent_escalation": _SHORT_PHRASE + " Examples: '3% annually', 'fixed steps per schedule', 'none'.",
        "expense_structure": _SHORT_PHRASE,
        "electricity_provision": "One short phrase built from the lease's own words: how electricity is supplied or "
                                 "metered, then who bills or pays and how, joined with 'and' (not commas). Name the "
                                 "billing party when the lease does, e.g. 'Separately metered and billed directly to "
                                 "Tenant by <utility company>', 'Separately metered and billed by Landlord as Additional "
                                 "Rent', 'Furnished by Landlord during ordinary business hours; excess electricity "
                                 "separately metered and paid by Tenant as additional rent'. When Landlord bills it, write 'billed by "
                                 "Landlord as Additional Rent' (not 'billed by Landlord and paid by Tenant').",
        "right_to_sublet": "State the consent requirement as a positive phrase using the lease's own words, e.g. "
                           "'with the prior written consent of Landlord'. If the same clause adds a reasonableness "
                           "standard, append it as ', not unreasonably withheld'. Add a stated exception in parentheses.",
        "permitted_use": "Copy the permitted-use phrase verbatim as the use clause states it, i.e. the words after "
                         "'use and occupy the Premises for (the)' (e.g. 'operation of law office or other general "
                         "office purpose', 'general office purposes'), dropping only trailing restrictions such as "
                         "'and no other purpose'. If a reference page states the use (e.g. 'USE: General Office: "
                         "Architecture'), copy that.",
        "late_fee": "Copy VERBATIM, word for word, the lease's late-charge language: start at the amount itself "
                    "(e.g. 'the greater of: (a) $75.00, or (b) ...' or 'One Hundred Fifty and 00/100 Dollars ($150.00) "
                    "late charge and shall bear interest ...') and continue through any interest-rate language, stopping "
                    "before a saving proviso such as 'unless a lesser rate shall then be the maximum rate permissible "
                    "by law ...'. Omit only the lead-in before the amount ('will be subject to a', 'equal "
                    "to'). Do not shorten, reorder or paraphrase.",
        "holdover_rent": "Copy VERBATIM, word for word, the phrase stating the holdover rent: start at the multiplier "
                         "or percentage (e.g. 'double the Base Rent ...', '150% of the amount of ...') and continue to "
                         "the end of that rent description, keeping cross-references such as 'set forth in Paragraph 1 "
                         "hereof' and qualifiers such as 'prorated on a daily basis'. Stop before a defined-term label "
                         "like (the \"Holdover Rate\") and before a following 'for each month ...' clause. Write a "
                         "spelled-out percentage as its numeral only ('one hundred seventy-five percent (175%)' -> "
                         "'175%'). Do not shorten or paraphrase.",
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
