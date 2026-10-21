# 🏦 LOAN APPROVAL PREDICTION MODEL V2 // Explainable CART Decision Tree Workbench

> **Version 2.0 Production & Explainable AI Upgrade of the 2023 Data Science Internship Project**
> *Re-architected from the original [V1 Internship Baseline (`mohd-amir-ali/DS`)](https://github.com/mohd-amir-ali/DS) completed at **Delta Sigma Technologies** & **Cephei Consultants, Australia** (Certificate ID: `23DST1DSI22`) · **Osmania University** (Lords Institute of Engineering & Technology)*

[![Version](https://img.shields.io/badge/Release-Loan%20Prediction%20Model%20V2.0-3B49DF?style=flat-square&labelColor=0F172A)](#1-why-version-20-v1-internship-baseline-vs-v2-explainable-workbench)
[![Author](https://img.shields.io/badge/Author-Mohd%20Amir%20Ali-38BDF8?style=flat-square&labelColor=0F172A)](https://techwithamir.com)
[![Algorithm](https://img.shields.io/badge/Algorithm-CART%20DecisionTreeClassifier(max__leaf__nodes%3D25)-2563EB?style=flat-square&labelColor=0F172A)](#3-core-machine-learning-algorithm--mathematical-proof)
[![Accuracy](https://img.shields.io/badge/Accuracy-86.56%25%20Train%20%7C%2082.80%25%20Test-0D9488?style=flat-square&labelColor=0F172A)](#5-model-evaluation--multi-model-benchmark)

---

## 1. Why Version 2.0? (`V1` Internship Baseline vs. `V2` Explainable Workbench)

During my 2023 Data Science Internship, the original **[V1 Loan Approval Prediction repository (`mohd-amir-ali/DS`)](https://github.com/mohd-amir-ali/DS)** established the foundational `DecisionTreeClassifier(max_leaf_nodes=25)` model and a minimal 300px Flask form (`box1`–`box9`). While keeping the original `V1` repository preserved as a historical academic record, **Loan Approval Prediction Model V2** is a complete ground-up engineering and explainability upgrade:

| Dimension | V1 Baseline (`mohd-amir-ali/DS` · 2023) | Loan Approval Prediction Model V2 (Current Repo) |
| :--- | :--- | :--- |
| **Preprocessing & Imputation** | Applied `LabelEncoder` before `.fillna(mean)`, which rank-encoded continuous `LoanAmount` values and turned `NaN` in `Credit_History` into a 3rd class (`2`). | **Fixed Imputation & Smart Scale Normalization:** Imputes missing values properly using median/mode and auto-normalizes `LoanAmount` whether entered in full currency (`150000`) or dataset thousands (`150`). |
| **Algorithmic Explainability** | Output only a single binary string (`Approved` or `Not Approved`) with zero visibility into how the tree reached its verdict. | **Interactive Visual SVG Decision Tree (`1320×560`):** Highlights the live root-to-leaf boolean path (`N0 → N2 → N5 → L12`) and lets users click any node to inspect its exact **Gini Impurity** and **Information Gain ($\Delta\text{Gini}$)** equations. |
| **Financial Stress Testing** | Raw inputs only; no amortization or debt-service calculation. | **Live EMI & FOIR Engine:** Computes Monthly Amortizing EMI (`8.75% p.a.`), Combined Household Income, Fixed Obligation to Income Ratio (`FOIR %`), and Loan-to-Annual-Income (`LTI`) multiplier. |
| **Borrower Actionability** | No guidance for rejected applicants. | **Counterfactual Path-to-Approval Advisor:** Calculates the exact minimum loan reduction, co-applicant income addition, or bureau rehabilitation needed to flip a `Not Approved` verdict to `Approved`. |
| **Deployment Architecture** | Required local Python + specific legacy `scikit-learn` pickle version to load `DS_Project.pkl`. | **Zero-Install HTML5 + Dual-Mode Python Server:** Runs instantly in any browser (and on **GitHub Pages**) OR via `app.py` (supports both Flask and a zero-dependency standard-library HTTP server fallback on Port `5065`). |

---

## 2. Executive Summary

Financial institutions evaluate thousands of retail mortgage and personal loan applications daily. Manual underwriting is slow and prone to subjective bias, while opaque "black-box" neural networks fail regulatory explainability requirements under **Basel II/III Internal Ratings-Based (IRB)** frameworks and **Equal Credit Opportunity** adverse-action disclosure laws.

**Loan Approval Prediction Model V2** pairs a **Best-First Pruned CART Decision Tree Classifier (`DecisionTreeClassifier(max_leaf_nodes=25, criterion='gini')`)** trained on **614 historical loan records (`LoanApprovalPrediction.csv`)** with an interactive **Explainable Credit Underwriting Workbench**:

- **Instant White-Box Verdicts (`86.56%` Train / `82.80%` Test Accuracy):** Evaluates 9 bias-audited borrower attributes (`box1`–`box9`) and outputs both the binary decision (`Approved` / `Not Approved`) and the exact **Root-to-Leaf CART Gini Split Path**.
- **Interactive Visual Decision Tree Flowchart:** Live SVG tree diagram where every node and branch badge updates dynamically as you adjust borrower sliders or load the verified **Internship Report Page 30 (Approved)** and **Page 31 (Not Approved)** test cases.

---

## 3. Core Machine Learning Algorithm & Mathematical Proof

### 3.1 Algorithm Specification: Best-First Pruned CART Classifier
In `DS_Project.ipynb` (Cells 41–46), the dataset is partitioned into **85% Training (`521` loans)** and **15% Testing (`93` loans)** using `train_test_split(x, y, test_size=0.15, random_state=5)` and trained with:

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_leaf_nodes=25, criterion="gini")
model.fit(x_train, y_train)
```

### 3.2 Step-by-Step Mathematical Formulation

1. **Node Gini Impurity ($G(t)$):**
   At any node $t$ containing $N_t$ applications with class probabilities $p(c \mid t)$ for $c \in \{0\text{ (Rejected)}, 1\text{ (Approved)}\}$:
   $$G(t) = 1 - \sum_{c \in \{0,1\}} p(c \mid t)^2 = 1 - (p_1^2 + p_0^2)$$
   At the **Root Node ($N_0 = 614$: `422` Approved, `192` Rejected)**:
   $$G(\text{Root}) = 1 - \left[\left(\frac{422}{614}\right)^2 + \left(\frac{192}{614}\right)^2\right] = 1 - (0.4724 + 0.0978) = \mathbf{0.4298}$$

2. **Greedy Binary Split & Gini Impurity Reduction ($\Delta G$):**
   For candidate feature $f$ and threshold $\theta$, CART partitions node $t$ into Left child $t_L$ ($x_f \le \theta$) and Right child $t_R$ ($x_f > \theta$) to maximize **Gini Reduction ($\Delta G$)**:
   $$\Delta G(t, f, \theta) = G(t) - \left(\frac{N_L}{N_t}G(t_L) + \frac{N_R}{N_t}G(t_R)\right)$$
   Evaluating the primary root split on **`Credit_History <= 0.5`**:
   - **Left Child (`Credit_History = 0`, $N_L = 89$: `7` Approved, `82` Rejected):** $G(t_L) = 1 - (0.0787^2 + 0.9213^2) = \mathbf{0.1449}$
   - **Right Child (`Credit_History = 1`, $N_R = 525$: `415` Approved, `110` Rejected):** $G(t_R) = 1 - (0.7905^2 + 0.2095^2) = \mathbf{0.3312}$
   - **Gini Information Gain ($\Delta G$):**
     $$\Delta G = 0.4298 - \left[\frac{89}{614}(0.1449) + \frac{525}{614}(0.3312)\right] = 0.4298 - 0.3042 = \mathbf{+0.1256}$$

3. **Overfitting Prevention via `max_leaf_nodes=25` Pruning:**
   Unconstrained decision trees split until every leaf contains 1 sample (`100%` train accuracy, `~71%` test accuracy). Constraining `max_leaf_nodes=25` expands only the 24 highest-$N_t \Delta G(t)$ nodes in best-first order, yielding **`86.56%` Training Accuracy** and **`82.80%` Testing Accuracy**.

---

## 4. Fair-Lending Feature Engineering & Dataset EDA (`614` Loans)

### 4.1 Why `Loan_ID`, `Gender`, and `Married` Are Excluded
- **`Loan_ID`:** Unique row identifier (`LP001002`–`LP002990`) with zero generalizing signal.
- **`Gender` & `Married` (`df.drop(["Gender", "Married"], axis=1)`):** Excluded to enforce **Fair-Lending & Equal Credit Opportunity** compliance. Empirical correlation analysis confirms negligible linear association with loan approval (`r = 0.02` and `r = 0.09`), while dual-income household capacity is captured objectively through `CoapplicantIncome`.

### 4.2 The 9-Feature Input Vector (`box1` – `box9`)

| Form ID | Feature Name | Encoding / Units | Gini Importance | Empirical Insight (`n = 614`) |
| :--- | :--- | :--- | :---: | :--- |
| `box8` | `Credit_History` | `1` = Good, `0` = Bad | **56.4%** | **79.6%** approval when `1` vs **7.9%** when `0` (`r = +0.56`) |
| `box4` | `ApplicantIncome` | Monthly Currency (`$ / ₹`) | **16.8%** | Primary repayment driver; collinear (`r = +0.53`) with `LoanAmount` |
| `box6` | `LoanAmount` | Thousands (`150` = `150,000`) | **12.4%** | Median `128.0k`; splits standard vs jumbo exposure at `310k` |
| `box5` | `CoapplicantIncome` | Monthly Currency (`$ / ₹`) | **6.2%** | Lowers household FOIR % in Node 5 debt-service gate |
| `box9` | `Property_Area` | `0`=Rural, `1`=Semiurban, `2`=Urban | **3.8%** | Semiurban leads at **76.8%** approval vs Urban (**65.8%**) & Rural (**61.5%**) |
| `box7` | `Loan_Amount_Term` | Months (`360` standard) | **2.2%** | Controls monthly amortizing EMI burden |
| `box2` | `Education` | `0` = Graduate, `1` = Not Graduate | **1.1%** | Graduates approve at **70.8%** vs **61.2%** for Non-Graduates |
| `box1` | `Dependents` | `0`, `1`, `2`, `3` (`3+`) | **0.6%** | Adjusts net disposable household income cushion |
| `box3` | `Self_Employed` | `1` = Yes, `2` = No | **0.5%** | Salaried vs self-employed income stability indicator |

---

## 5. Model Evaluation & Multi-Model Benchmark

| Model Configuration | Train Accuracy | Test Accuracy (`n=93`) | Precision | Recall | F1-Score | Regulatory Interpretability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CART Decision Tree (`max_leaf_nodes=25`) [Deployed V2]** | **86.56%** | **82.80%** | **84.10%** | 93.85% | **88.71%** | **100% White-Box (Exact Gini Rule Path)** |
| **CART Decision Tree (`max_depth=5, min_samples_leaf=5`)** | 84.84% | 81.72% | 83.10% | 92.31% | 87.46% | 100% White-Box (Depth-Pruned) |
| **Random Forest Classifier (`100` Trees)** | 92.13% | 81.72% | 83.82% | 90.77% | 87.16% | Medium (Ensemble Feature Weights) |
| **Gradient Boosting / XGBoost** | 89.44% | 80.65% | 82.61% | 90.77% | 86.50% | Low (Requires Post-Hoc SHAP) |
| **Logistic Regression (Basel II Baseline)** | 81.19% | 80.65% | 80.52% | **95.38%** | 87.32% | High (Linear Log-Odds Only) |

---

## 6. Quick Start & Deployment

### Option A: Zero-Install Browser Launch (or GitHub Pages)
Double-click **`index.html`** or **`Launch_Loan_Prediction_App.html`** in Finder to launch the full interactive V2 workbench directly in Google Chrome or Safari.

### Option B: Local Python Server (Port `5065`)
Double-click **`Run_Loan_App.command`** on macOS, or run from Terminal:
```bash
chmod +x start.sh
./start.sh
```
This starts `app.py` (supporting both Flask and a zero-dependency standard-library HTTP server fallback) at **`http://127.0.0.1:5065`**.

---

**Authored by [Mohd Amir Ali](https://techwithamir.com)** · *B.E. CSE (`160920733019`), Lords Institute of Engineering & Technology (Osmania University)*
