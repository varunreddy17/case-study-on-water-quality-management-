# Smart Health Surveillance & Early Warning (ML) 🚑📊

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-lightgrey?logo=pandas&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)

A **machine learning-based system** for predicting multiple diseases and assessing outbreak risk using synthetic health and environmental data.

---

## Features ✨

- **Multi-label Disease Prediction**  
  Predicts multiple co-occurring diseases (malaria, dengue, cholera, flu) using environmental and temporal data.

- **Temporal & Seasonal Feature Engineering**  
  - Lag features (previous day rainfall/temperature)  
  - Moving averages (7-day rolling averages)  
  - One-hot encoding for districts

- **Outbreak Risk Scoring / Early Warning**  
  - Predicts probability of any disease outbreak  
  - Provides sample probabilities and AUC score  

---

## Requirements 🛠️

- Python 3.8+  
- Libraries: `numpy`, `pandas`, `scikit-learn`

Install dependencies:

```bash
pip install numpy pandas scikit-learn
