APPEAL_SNIPPETS = {
    "missing_auth": (
        "Appeal Strategy for Missing Prior Authorization: Submit proof of emergency medical necessity "
        "or retro-authorization documentation. Attach medical records demonstrating urgent patient care "
        "requirements that precluded advance authorization."
    ),
    "eligibility_lapse": (
        "Appeal Strategy for Eligibility Lapse: Provide member enrollment history and employer policy "
        "effective date confirmation. Include proof of active coverage on service date and secondary payer coordination."
    ),
    "duplicate_claim": (
        "Appeal Strategy for Duplicate Claim Denial: Clarify distinct clinical services, multiple anatomical "
        "sites, or separate encounter times. Attach modifier documentation (-59 or -76) and itemized billing statements."
    ),
    "coding_error": (
        "Appeal Strategy for Coding / Bundling Denial: Provide physician progress notes, CPT descriptor mapping, "
        "and ICD-10 medical necessity justification demonstrating independent clinical procedural value."
    ),
    "out_of_network": (
        "Appeal Strategy for Out-of-Network Service: Highlight network inadequacy, emergency stabilization requirements, "
        "or prior approved referral documentation under ACA continuity of care rules."
    ),
    "experimental_procedure": (
        "Appeal Strategy for Experimental/Investigational Denial: Include peer-reviewed medical journal evidence, "
        "NCCN/FDA guideline approvals, and physician attestation of standard-of-care clinical necessity."
    ),
    "timely_filing": (
        "Appeal Strategy for Timely Filing Denial: Attach EDI clearinghouse acceptance report, certified mail "
        "delivery proof, or primary payer Explanation of Benefits (EOB) demonstrating timely initial submission."
    )
}

DEFAULT_SNIPPET = (
    "General Appeal Strategy: Review specific denial code on EOB/RA. Assemble physician progress notes, "
    "itemized claim breakdown, and formal letter detailing medical necessity under payer clinical policy guidelines."
)

def retrieve_appeal_guidance(denial_reason: str) -> dict:
    """
    Retrieves relevant appeal guidance snippet based on keyword matching.
    """
    if not denial_reason:
        return {"matched_reason": "default", "snippet": DEFAULT_SNIPPET}
        
    reason_clean = denial_reason.lower().strip()
    
    # Exact or keyword matching
    for key, snippet in APPEAL_SNIPPETS.items():
        if key in reason_clean or any(word in reason_clean for word in key.split('_')):
            return {
                "matched_reason": key,
                "snippet": snippet
            }
            
    return {
        "matched_reason": "general",
        "snippet": DEFAULT_SNIPPET
    }

LLM_HALLUCINATION_MITIGATION_RULES = [
    "1. Strict Grounding Prompting: Use system prompts requiring the LLM to rely ONLY on retrieved EOB policy snippets.",
    "2. Structured JSON Output Schema: Enforce Pydantic / JSON schema for output to prevent narrative drift.",
    "3. Mandatory Source Citation: Require the model to explicitly quote source clause IDs for every clinical assertion.",
    "4. Variable Masking & Guardrails: Inject verified patient/billing metadata as non-modifiable template variables.",
    "5. Output Verification & Fallback: Validate output against regex rules for dollar amounts and dates before dispatching."
]
