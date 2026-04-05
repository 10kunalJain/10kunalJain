# Kunal Jain

**ML Engineer** — I build production-grade ML systems, not just models. My projects have modular codebases, automated test suites, Docker deployments, CI/CD pipelines, and honest evaluation with failure analysis.

---

### Featured Projects

#### [Personalized Fashion Recommendation System](https://github.com/10kunalJain/recommendation-system)
Two-stage recommendation pipeline (retrieval + ranking) processing 197K transactions across 19.7K items. 5 retrieval models (ALS, Two-Tower neural, content-based, recency, popularity) fused via reciprocal rank fusion, re-ranked by LightGBM LambdaRank. **1.7x MAP@12 improvement** over best baseline. FastAPI serving at P50=43ms latency.

`PyTorch` `LightGBM` `FastAPI` `implicit` `scikit-learn` `Docker` `GitHub Actions CI`

- 49 unit tests | Dockerfile + docker-compose | CI/CD pipeline | Segment-wise evaluation with bootstrap CIs

#### [Real-Time Driver Drowsiness Detection](https://github.com/10kunalJain/drowsiness-detection) | [Live Demo](https://drowsiness-detection-fwufwjunwv6eupgxfvcw3w.streamlit.app/)
End-to-end CV system: ResNet50V2 + LSTM temporal head + uncertainty estimation + 4-state fatigue machine. Robustness-tested across 36 corruption conditions. Error-driven improvement loop pushed AUC from 0.902 to **0.988**. LSTM sequence accuracy: **96.3%**. Deployed on Streamlit Cloud.

`TensorFlow` `Keras` `OpenCV` `TFLite` `Streamlit` `Grad-CAM`

- Robustness testing (6 corruptions x 6 severities) | TFLite edge export (23MB) | Uncertainty-aware predictions

---

### How I Build

| Principle | How I Apply It |
|:---|:---|
| **System design > single models** | Two-stage retrieval + ranking pipeline; multi-model fusion; cold-start fallbacks |
| **Evaluation rigor** | Temporal splits (no leakage), baseline comparisons, segment-wise metrics with confidence intervals |
| **Failure-aware engineering** | Robustness testing, failure case analysis, honest "why metrics are low" documentation |
| **Production mindset** | Docker, CI/CD, FastAPI serving, latency profiling, health checks |
| **Tested code** | 49+ unit tests, automated linting, coverage reporting |

---

### Tech Stack

**ML/DL:** PyTorch, TensorFlow, LightGBM, scikit-learn, implicit, NumPy, Pandas

**Serving:** FastAPI, Streamlit, Docker, GitHub Actions

**Techniques:** Two-Tower retrieval, LambdaRank, ALS (implicit feedback), Reciprocal Rank Fusion, Transfer Learning, LSTM, Grad-CAM, Test-Time Augmentation, Bootstrap CI evaluation

---

[LinkedIn](https://www.linkedin.com/in/kunal-jain-27b4b0209) | 10.kunaljain@gmail.com
