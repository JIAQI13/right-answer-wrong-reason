#!/usr/bin/env python
"""CLI wrapper for activation patching analysis (H2 - causal evidence).

Runs selective patching on representative cases:
  - 5 sycophantic_failure (model fell for the hint)
  - 5 right_answer_wrong_reason (RAWR cases)
  - 5 faithful_reasoning (control)

Why selective, not full dataset:
  - Full patching = ~20K generations (expensive)
  - 15 cases × 3 layers = 45 generations (practical)
  - Focus on cases where we already see behavioral differences
"""
from rawr.patching_analysis import main

if __name__ == "__main__":
    main()
