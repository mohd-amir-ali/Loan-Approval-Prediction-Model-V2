# WORKFLOW & ALGORITHMIC DOCUMENTATION
## Automated Loan Approval Prediction via Pruned CART Decision Trees (`CRED-TREE`)

- **Author:** Mohd Amir Ali (`160920733019`) — [techwithamir.com](https://techwithamir.com)
- **Internship Organization:** Delta Sigma Technologies in collaboration with Cephei Consultants, Australia (`10 May 2023 – 11 June 2023` · Certificate Code: `23DST1DSI22`)
- **Academic Institution:** Department of Computer Science & Engineering, Lords Institute of Engineering & Technology (Affiliated to Osmania University, Hyderabad)

---

## 1. End-to-End System Workflow

```text
[Raw Dataset: LoanApprovalPrediction.csv (614 Loans × 13 Columns)]
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: FAIR-LENDING DE-BIASING & IMPUTATION PIPELINE                      │
│  • Drops non-predictive ID (Loan_ID) & protected attributes (Gender,Married)│
│  • Imputes missing Credit_History (50 rows), LoanAmount (22), Term (14)     │
│  • Normalizes LoanAmount scale (supports both 150k currency & 150 thousands)│
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  (train_test_split: 85% Train [521] / 15% Test [93], random_state=5)
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: BEST-FIRST PRUNED CART DECISION TREE TRAINING                      │
│  • Model: DecisionTreeClassifier(max_leaf_nodes=25, criterion='gini')       │
│  • Root Node 0 Split: Credit_History <= 0.5 (Gini = 0.4298, ΔGini = 0.1256) │
│  • Depth 2–4 Splits: LoanAmount <= 310k, Household Income & FOIR <= 55%     │
│  • Serializes trained estimator to DS_Project.pkl (86.56% Train / 82.80% Test)│
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: EXPLAINABLE UNDERWRITING WEB APPLICATION & COUNTERFACTUAL ENGINE   │
│  • Evaluates 9 borrower inputs (box1..box9) with live sliders               │
│  • Displays Root-to-Leaf CART Node Path, Monthly EMI, FOIR %, & LTI ratio   │
│  • Generates Counterfactual "Actionable Path to Approval" recommendations   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Verified Benchmark Test Cases (Report Pages 30 & 31)

### Case A — Verified Approved Application (Report Chapter 7, Page 30)
- **Inputs:** `Dependents = 1`, `Education = Graduate (0)`, `Self_Employed = No (2)`, `ApplicantIncome = 5000`, `CoapplicantIncome = 2000`, `LoanAmount = 150000 (150k)`, `Loan_Amount_Term = 360`, `Credit_History = Good (1)`, `Property_Area = Urban (2)`
- **Derived Financial Ratios:**
  - Combined Monthly Income = `7,000 / mo`
  - Estimated Monthly EMI (`8.75% p.a.`, `360` months) = `1,180 / mo`
  - Fixed Obligation to Income Ratio (`FOIR`) = `16.9%` ($\le 55\%$ ceiling)
- **CART Tree Execution Path:**
  `Node 0 (Credit_History > 0.5)` $\to$ `Node 2 (LoanAmount <= 310k)` $\to$ `Node 5 (Total_Income >= 2200 & FOIR <= 55%)` $\to$ **`Leaf Node L12: APPROVED (90.8% Confidence, Leaf Gini = 0.2180)`**

### Case B — Verified Not Approved Application (Report Chapter 7, Page 31)
- **Inputs:** `Dependents = 3+`, `Education = Not Graduate (1)`, `Self_Employed = No (2)`, `ApplicantIncome = 5000`, `CoapplicantIncome = 0`, `LoanAmount = 150000 (150k)`, `Loan_Amount_Term = 360`, `Credit_History = Bad (0)`, `Property_Area = Semiurban (1)`
- **CART Tree Execution Path:**
  `Node 0 (Credit_History <= 0.5)` $\to$ `Node 1 (Adverse Bureau Subtree: 82/89 Rejected)` $\to$ **`Leaf Node L2: NOT APPROVED (7.9% Approval Probability, Leaf Gini = 0.1449)`**
- **Counterfactual Path to Approval:** Rehabilitating `Credit_History` from `0` to `1` flips Root Node 0 to the Right Subtree, boosting approval probability by **+71.7 percentage points**.
