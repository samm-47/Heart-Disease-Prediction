import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree

# Load the dataset
# Ensure you have 'heart.csv' in the same folder
df = pd.read_csv('heart.csv')

# 1. Feature Selection
# Based on your heatmap, these are the strongest predictors
# But for now, let's use all features as RF handles them well
X = df.drop('target', axis=1)
y = df['target']

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
# We use a smaller max_depth to prevent overfitting on clinical data
rf_heart = RandomForestClassifier(
    n_estimators=100, 
    max_depth=7, 
    random_state=42,
    class_weight='balanced'
)
rf_heart.fit(X_train_scaled, y_train)

# 4. Predictions
y_pred = rf_heart.predict(X_test_scaled)

# 5. Results
print(f"Heart Disease Prediction Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


# Add this at the end of your script
importances = rf_heart.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("Which Health Metrics Matter Most?")
plt.show()

# Run 5-fold cross-validation
cv_scores = cross_val_score(rf_heart, X_train_scaled, y_train, cv=5)

print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")


# Select one tree from the forest
sub_tree = rf_heart.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(sub_tree, 
          feature_names=X.columns, 
          class_names=['Healthy', 'Ill'], 
          filled=True, 
          max_depth=3, # Limit depth so it's readable
          rounded=True, 
          fontsize=10)
plt.title("Decision Path: How the Model Diagnoses a Patient")
plt.show()


def predict_heart_risk(patient_data):
    # Ensure patient_data is a DataFrame with same columns as X
    patient_df = pd.DataFrame([patient_data])
    
    # Scale the input using the SAME scaler from training
    patient_scaled = scaler.transform(patient_df)
    
    # Get probability instead of just 0 or 1
    risk_proba = rf_heart.predict_proba(patient_scaled)[0][1]
    
    print(f"\n--- Diagnostic Result ---")
    print(f"Calculated Heart Disease Risk: {risk_proba:.2%}")
    if risk_proba > 0.5:
        print("Status: HIGH RISK - Clinical consultation recommended.")
    else:
        print("Status: LOW RISK")

# Example: Testing a high-risk scenario
# (Using values similar to your high-importance features: cp, ca, thal)
test_patient = {
    'age': 65, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233, 
    'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0, 
    'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}

predict_heart_risk(test_patient)


import joblib

# Save the model
joblib.dump(rf_heart, 'heart_disease_model.pkl')

# Save the scaler (essential for processing new patient data later)
joblib.dump(scaler, 'heart_scaler.pkl')

print("\nSuccess! Model and Scaler saved as .pkl files.")