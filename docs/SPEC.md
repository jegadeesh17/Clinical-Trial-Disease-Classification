# Clinical Trial Disease Category Classification — Technical Specification

---

## 1. Document Control & System Overview

| Field | Specification |
| :--- | :--- |
| **System** | Clinical Trial Disease Category Classification Engine |
| **Document** | `docs/SPEC.md` |
| **Version** | 2.0.0 (Production Grade) |
| **Domain** | Healthcare NLP / Clinical Protocol Triage |
| **Maintainer** | Applied AI / Healthcare NLP Engineering |

### 1.1 Problem Definition & Clinical Context
Clinical trial registries receive tens of thousands of unstructured trial protocols with heterogeneous medical terminologies. Automated categorization of trial eligibility and disease indication accelerates patient cohort matching, regulatory compliance reviews, and therapeutic pipeline tracking across 8 major disease classes:
`anxiety`, `breast cancer`, `chronic obstructive pulmonary disease`, `covid-19`, `glaucoma`, `rheumatoid arthritis`, `sickle cell anemia`, and `type 2 diabetes`.

---

## 2. System Architecture & NLP Pipeline

```mermaid
flowchart TD
    A[Unstructured Clinical Trial Summary / Protocol] --> B[FastAPI Gateway: POST /predict]
    B --> C[Medical Text Preprocessing & Lemmatization]
    C --> D[Stopword Removal & Regex Normalization]
    D --> E[Sublinear TF-IDF Vectorizer (max_features=15,000, ngram=(1,2))]
    E --> F[Calibrated Logistic Regression / Random Forest]
    F --> G[Predict Probability Distribution across 8 Disease Categories]
    G --> H[Top Category & Confidence Margin Response]
```

---

## 3. Latency & Performance Budgets

| Metric | Target Specification |
| :--- | :--- |
| **p50 Latency (CPU)** | $\le 10\text{ ms}$ |
| **p95 Latency (CPU)** | $\le 25\text{ ms}$ |
| **Memory Footprint** | $\le 180\text{ MB RAM}$ |
| **Model Size** | $1.59\text{ MB}$ (`classifier_pipeline_best.joblib`) |
| **Throughput Target** | $\ge 150\text{ requests/sec}$ |

---

## 4. API Specification

### `GET /health`
Returns pipeline artifact availability and active classes list.

### `POST /predict`
* **Input**:
```json
{
  "summary": "Phase III study evaluating the efficacy of monoclonal antibodies in patients with metastatic HER2-positive carcinoma.",
  "top_k": 3
}
```
* **Output**:
```json
{
  "predicted_category": "breast cancer",
  "confidence": 0.9412,
  "top_predictions": [
    {"category": "breast cancer", "probability": 0.9412},
    {"category": "type 2 diabetes", "probability": 0.0182},
    {"category": "rheumatoid arthritis", "probability": 0.0124}
  ],
  "inference_time_ms": 11.4
}
```

---

## 5. Failure Modes & Mitigations

1. **Short / Noisy Text**: Preprocessor safely handles empty strings, returns a fallback low-confidence general flag rather than throwing an exception.
2. **Container Security**: Multi-stage build with non-root user `appuser` (UID 10001).
