# AI Ethics & Compliance Report Draft

## Scope

This draft summarizes ethical and compliance risks observed in Greenwashing Detector outputs.

## Cases

### Case 1: Unknown claim

- Risk level: `HIGH`
- Risk categories: `absolute_claim, hidden_tradeoff, carbon_offset_or_net_zero_risk`
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: Our bottle is made from 100% recycled materials and achieves carbon neutrality through verified emissions reductions, with third-party verification of our environmental claims.

### Case 2: Unknown claim

- Risk level: `MEDIUM`
- Risk categories: `unsubstantiated_claim`
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: This package contains 60% recycled paper, based on supplier documentation. For full verification details, please contact [company] or visit [website].

### Case 3: Unknown claim

- Risk level: `INSUFFICIENT_EVIDENCE`
- Risk categories: ``
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: Evidence is insufficient. Provide specific environmental attributes, scope, and substantiation before making a claim.

### Case 4: Unknown claim

- Risk level: `HIGH`
- Risk categories: `recyclability_ambiguity, absolute_claim`
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: This product is recyclable in areas with appropriate recycling facilities. Recycling availability may vary by location.

### Case 5: Unknown claim

- Risk level: `HIGH`
- Risk categories: `future_target_overclaim, carbon_offset_or_net_zero_risk`
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: Our company is committed to achieving net-zero emissions by 2030, and we are actively working to reduce the environmental impact of our products. While we are making progress toward this goal, our current products are not yet climate-friendly, and we are continuously improving our sustainability practices.

### Case 6: Unknown claim

- Risk level: `HIGH`
- Risk categories: `vague_general_claim, unsubstantiated_claim`
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: Reduced environmental impact compared to previous version, with 30% lower carbon emissions verified through lifecycle analysis.

### Case 7: Unknown claim

- Risk level: `INSUFFICIENT_EVIDENCE`
- Risk categories: ``
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: Evidence is insufficient. Provide specific environmental attributes, scope, and substantiation before making a claim.

### Case 8: Unknown claim

- Risk level: `HIGH`
- Risk categories: `unsubstantiated_claim, misleading_comparison`
- Citation audit passes: `True`
- Ethics audit passes: `True`
- Responsible rewrite: Our refill pouch uses 40% less plastic than our previous bottle, based on [specific metric or data source].

## Mitigation Principles

1. Use risk-indication language, not final legal accusations.
2. Mark evidence gaps clearly.
3. Ground every risk category in cited guidance or evidence.
4. Avoid vague green wording in rewrites.
5. Keep consumer explanations neutral and educational.
