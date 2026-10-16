# 🏦 CRED-TREE // Explainable CART Decision Tree Loan Approval & Credit Risk Engine

> **Data Science Internship Project — Automated Retail Loan Underwriting, Root-to-Leaf Gini Path Tracing & Fair-Lending Audit**
> *Completed at **Delta Sigma Technologies** in collaboration with **Cephei Consultants, Australia** (Certificate ID: `23DST1DSI22`) · Submitted to **Osmania University** (Lords Institute of Engineering & Technology)*

[![Author](https://img.shields.io/badge/Author-Mohd%20Amir%20Ali-38BDF8?style=flat-square&labelColor=0B132B)](https://techwithamir.com)
[![Algorithm](https://img.shields.io/badge/Algorithm-CART%20DecisionTreeClassifier(max__leaf__nodes%3D25)-2563EB?style=flat-square&labelColor=0B132B)](#2-core-machine-learning-algorithm--mathematical-proof)
[![Accuracy](https://img.shields.io/badge/Accuracy-86.56%25%20Train%20%7C%2082.80%25%20Test-10B981?style=flat-square&labelColor=0B132B)](#4-model-evaluation--multi-model-benchmark)
[![Compliance](https://img.shields.io/badge/Fair%20Lending-Gender%20%26%20Marital%20Bias%20Excluded-F59E0B?style=flat-square&labelColor=0B132B)](#3-fair-lending-feature-engineering--dataset-eda-614-loans)

---

## 1. Executive Summary

Financial institutions evaluate thousands of retail mortgage and personal loan applications daily. Manual underwriting is slow and prone to subjective bias, while opaque "black-box" neural networks fail regulatory explainability requirements under **Basel II/III Internal Ratings-Based (IRB)** frameworks and **Equal Credit Opportunity** adverse-action disclosure laws.

**CRED-TREE** solves this by pairing a **Best-First Pruned CART Decision Tree Classifier (`DecisionTreeClassifier(max_leaf_nodes=25, criterion='gini')`)** trained on **614 historical loan records (`LoanApprovalPrediction.csv`)** with an interactive **Explainable Credit Underwriting Web Application**:

- **Instant White-Box Verdicts (`86.56%` Train / `82.80%` Test Accuracy):** Evaluates 9 bias-audited borrower attributes (`box1`–`box9`) and outputs both the binary decision (`Approved` / `Not Approved`) and the exact **Root-to-Leaf CART Gini Split Path**.
- **Real-Time Financial Stress Testing:** Computes **Monthly Amortizing EMI** (at `8.75% p.a.`), **Combined Household Income**, **Fixed Obligation to Income Ratio (FOIR %)**, and **Loan-to-Annual-Income Multiplier (LTI)**.
- **Counterfactual "Actionable Path to Approval" Advisor:** Automatically calculates the minimum adjustments (e.g., bureau rehabilitation, loan principal reduction, or co-applicant income addition) required for a rejected borrower to cross the tree's approval boundary.

---

## 2. Core Machine Learning Algorithm & Mathematical Proof

### 2.1 Algorithm Specification: Best-First Pruned CART Classifier
In `DS_Project.ipynb` (Cells 41–46), the dataset is partitioned into **85% Training (`521` loans)** and **15% Testing (`93` loans)** using `train_test_split(x, y, test_size=0.15, random_state=5)` and trained with:

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_leaf_nodes=25, criterion="gini")
model.fit(x_train, y_train)
```

### 2.2 Step-by-Step Mathematical Formulation

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

## 3. Fair-Lending Feature Engineering & Dataset EDA (`614` Loans)

### 3.1 Why `Loan_ID`, `Gender`, and `Married` Are Excluded
- **`Loan_ID`:** Unique row identifier (`LP001002`–`LP002990`) with zero generalizing signal.
- **`Gender` & `Married` (`df.drop(["Gender", "Married"], axis=1)`):** Excluded to enforce **Fair-Lending & Equal Credit Opportunity** compliance. Empirical correlation analysis confirms negligible linear association with loan approval (`r = 0.02` and `r = 0.09`), while dual-income household capacity is captured objectively through `CoapplicantIncome`.

### 3.2 The 9-Feature Input Vector (`box1` – `box9`)

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

## 4. Model Evaluation & Multi-Model Benchmark

| Model Configuration | Train Accuracy | Test Accuracy (`n=93`) | Precision | Recall | F1-Score | Regulatory Interpretability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CART Decision Tree (`max_leaf_nodes=25`) [Deployed]** | **86.56%** | **82.80%** | **84.10%** | 93.85% | **88.71%** | **100% White-Box (Exact Gini Rule Path)** |
| **CART Decision Tree (`max_depth=5, min_samples_leaf=5`)** | 84.84% | 81.72% | 83.10% | 92.31% | 87.46% | 100% White-Box (Depth-Pruned) |
| **Random Forest Classifier (`100` Trees)** | 92.13% | 81.72% | 83.82% | 90.77% | 87.16% | Medium (Ensemble Feature Weights) |
| **Gradient Boosting / XGBoost** | 89.44% | 80.65% | 82.61% | 90.77% | 86.50% | Low (Requires Post-Hoc SHAP) |
| **Logistic Regression (Basel II Baseline)** | 81.19% | 80.65% | 80.52% | **95.38%** | 87.32% | High (Linear Log-Odds Only) |

---

## 5. Quick Start & Deployment

### Option A: Zero-Install Browser Launch (or GitHub Pages)
Double-click **`index.html`** or **`Launch_Loan_Prediction_App.html`** in Finder to launch the full 4-tab interactive underwriting terminal directly in Google Chrome or Safari.

### Option B: Local Python Server (Port `5065`)
Double-click **`Run_Loan_App.command`** on macOS, or run from Terminal:
```bash
chmod +x start.sh
./start.sh
```
This starts `app.py` (supporting both Flask and a zero-dependency standard-library HTTP server fallback) at **`http://127.0.0.1:5065`**.

### Repository Structure
```text
DS-main/
├── index.html                        # Primary 4-tab interactive underwriting platform (GitHub Pages ready)
├── Launch_Loan_Prediction_App.html   # Desktop standalone quick-launch mirror
├── Printable_Documentation.html      # Print / Save-as-PDF Technical Monograph & Workflow Report
├── WORKFLOW_DOCUMENTATION.md         # Detailed step-by-step system workflow & mathematical proof
├── README.md                         # Project overview & quick-start guide
├── app.py                            # Dual-mode Flask + zero-dependency HTTP server (Port 5065)
├── loan_cart_engine.py               # Pure-Python + Sklearn CART Decision Tree & Counterfactual Engine
├── DS_Project.ipynb                  # Original Data Science Internship Jupyter Notebook
├── DS_Project.pkl                    # Serialized DecisionTreeClassifier(max_leaf_nodes=25) artifact
├── LoanApprovalPrediction.csv        # 614-row historical retail loan dataset
├── Run_Loan_App.command              # One-click macOS launcher
├── start.sh                          # Terminal launcher script
├── static/styles/style.css           # Institutional FinTech Slate Navy & Royal Blue UI stylesheet
└── templates/index.html              # Flask template entry point
```

---

**Authored by [Mohd Amir Ali](https://techwithamir.com)** · *B.E. CSE (`160920733019`), Lords Institute of Engineering & Technology (Osmania University)*
