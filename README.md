# Heart Disease Diagnostic Classifier

An end-to-end Machine Learning pipeline and web application designed to predict the likelihood of heart disease in patients based on clinical health metrics.

### Key Results:
* **Model Accuracy:** 96.59%
* **Cross-Validation Score:** 96.46% (± 2.7%)
* **Primary Predictors:** Chest Pain Type (cp), Number of Major Vessels (ca), and Thalassemia (thal).

## Tech Stack
* **Language:** Python 3.x
* **Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib
* **Deployment:** Streamlit (Web Interface), Joblib (Model Persistence)

## Engineering Highlights
1. **Feature Signal Analysis:** Conducted correlation mapping to filter low-impact features (like Fasting Blood Sugar) and prioritize high-impact biomarkers.
2. **Model Stability:** Utilized 5-fold cross-validation to ensure the model generalizes well to unseen patient data.
3. **Interpretability:** Generated decision tree visualizations to map the logic used by the ensemble model for clinical transparency.
4. **Diagnostic Tool:** Built a custom simulation function that outputs a probability percentage rather than a simple binary classification.
