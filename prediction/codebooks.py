"""Intervention codebooks for dashboard runtime scenarios."""

INTERVENTION_CODEBOOK = {
    "reduce_ahs": {
        "var": "stigma_ahs",
        "delta": -0.5,
        "effect": "probability_relative_to_baseline",
        "label": "Reduce anticipated healthcare stigma by 50%",
    },
    "reduce_gss": {
        "var": "stigma_gss",
        "delta": -0.5,
        "effect": "probability_relative_to_baseline",
        "label": "Reduce general social stigma by 50%",
    },
    "reduce_family_stigma": {
        "var": "stigma_family",
        "delta": -0.5,
        "effect": "probability_relative_to_baseline",
        "label": "Reduce family stigma by 50%",
    },
}

REL_CODEBOOK = {
    "weaken_ahs_to_prep": {
        "from": "stigma_ahs",
        "to": "prep_used",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken AHS -> PrEP pathway by 50%",
    },
    "weaken_ahs_to_disclosure": {
        "from": "stigma_ahs",
        "to": "out_gid",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken AHS -> outness pathway by 50%",
    },
    "strengthen_outness_to_prep": {
        "from": "out_gid",
        "to": "prep_used",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen outness -> PrEP linkage by 50%",
    },
    "strengthen_outness_to_hivtest": {
        "from": "out_gid",
        "to": "hivtest12",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen outness -> HIV testing linkage by 50%",
    },
    "strengthen_seehcp_to_hivtest": {
        "from": "seehcp",
        "to": "hivtest12",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen healthcare -> HIV testing linkage by 50%",
    },
    "weaken_ahs_to_hivtest": {
        "from": "stigma_ahs",
        "to": "hivtest12",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken AHS -> HIV testing pathway by 50%",
    },
    "strengthen_seehcp_to_lower_ahs": {
        "from": "seehcp",
        "to": "stigma_ahs",
        "delta": 0.5,
        "effect": "multiplicative",
        "label": "Strengthen healthcare -> lower AHS pathway by 50%",
    },
    "weaken_outness_to_ahs_feedback": {
        "from": "out_gid",
        "to": "stigma_ahs",
        "delta": 0.5,
        "effect": "attenuate",
        "label": "Weaken outness -> AHS stigma feedback by 50%",
    },
}
