# WORKFLOW & ALGORITHMIC DOCUMENTATION
## LOAN APPROVAL PREDICTION MODEL V2 — Explainable CART Decision Tree Workbench

- **Author:** Mohd Amir Ali (`160920733019`) — [techwithamir.com](https://techwithamir.com)
- **Version Lineage:** Upgraded **V2.0 Explainable AI Release** built upon the [2023 V1 Internship Repository (`mohd-amir-ali/DS`)](https://github.com/mohd-amir-ali/DS)
- **Internship Organization:** Delta Sigma Technologies in collaboration with Cephei Consultants, Australia (`10 May 2023 – 11 June 2023` · Certificate Code: `23DST1DSI22`)
- **Academic Institution:** Department of Computer Science & Engineering, Lords Institute of Engineering & Technology (Affiliated to Osmania University, Hyderabad)

---

## 1. Evolution from V1 (2023) to V2.0 (Explainable CART Workbench)

1. **Preserving the V1 Baseline (`mohd-amir-ali/DS`):** The original 2023 repository remains intact as the historical internship submission archive.
2. **What V2.0 Adds:**
   - **Interactive SVG Decision Tree Visualizer (`1320×560`):** Displays every split node (`N0`, `N1`, `N2`, `N5`, `N6`) and leaf (`L1`, `L2`, `L12`, `L15`, `L19`, `L20`) with opaque branch badges, live path highlighting, and an interactive click inspector showing exact **Gini Impurity** and **Information Gain ($\Delta\text{Gini}$)** math.
   - **Live Financial Underwriting Telemetry:** Calculates Monthly Amortizing EMI (`8.75% p.a.`), Combined Household Income, Fixed Obligation to Income Ratio (`FOIR %`), and Loan-to-Annual-Income (`LTI`) multiplier.
   - **Counterfactual Approval Advisor:** Tells rejected borrowers the exact minimum change needed to qualify.
   - **Dual-Mode Zero-Dependency Runtime:** Works both as a zero-install HTML5 / GitHub Pages app (`index.html`) and via `app.py` on Port `5065`.

---

## 2. End-to-End V2 System Workflow

```text
[Raw Dataset: LoanApprovalPrediction.csv (614 Loans × 13 Columns)]
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: FAIR-LENDING DE-BIASING & V2 IMPUTATION PIPELINE                   │
│  • Drops non-predictive ID (Loan_ID) & protected attributes (Gender,Married)│
│  • Imputes missing Credit_History (50 rows), LoanAmount (22), Term (14)     │
│  • Auto-normalizes LoanAmount (supports both 150000 currency & 150k units)  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  (train_test_split: 85% Train [521] / 15% Test [93], random_state=5)
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: BEST-FIRST PRUNED CART DECISION TREE TRAINING                      │
│  • Model: DecisionTreeClassifier(max_leaf_nodes=25, criterion='gini')       │
│  • Root Node 0 Split: Credit_History <= 0.5 (Gini = 0.4298, ΔGini = 0.1256) │
│  • Depth 2–4 Splits: LoanAmount <= 310k, Household Income & FOIR <= 55%     │
│  • Verified Accuracy: 86.56% Training | 82.80% Testing                      │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: V2 EXPLAINABLE UNDERWRITING WORKBENCH & LIVE SVG TREE VISUALIZER   │
│  • Evaluates 9 borrower inputs (box1..box9) with live sliders & presets     │
│  • Highlights active Root-to-Leaf path on the 1320×560 SVG Decision Tree    │
│  • Generates Counterfactual "Actionable Path to Approval" recommendations   │
└─────────────────────────────────────────────────────────────────────────────┘
```
