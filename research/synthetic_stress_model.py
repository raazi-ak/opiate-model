# Synthetic Stress Detection Model
# This script creates a realistic synthetic dataset for stress detection using physiological signals and builds an explainable model.

# Install required libraries
# !pip install numpy pandas scikit-learn matplotlib seaborn shap plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)
plt.style.use('seaborn-v0_8')

def generate_synthetic_stress_data(n_samples=10000, n_subjects=50):
    """
    Generate realistic synthetic physiological data for stress detection.
    
    Stress levels:
    0: Low stress (baseline)
    1: Medium stress 
    2: High stress
    """
    
    data = []
    
    for subject_id in range(n_subjects):
        # Subject-specific baseline characteristics
        baseline_hr = np.random.normal(72, 8)  # Baseline heart rate
        baseline_hrv = np.random.normal(45, 12)  # Baseline HRV (RMSSD)
        baseline_movement = np.random.normal(0.5, 0.2)  # Baseline movement
        
        # Generate samples for this subject
        for _ in range(n_samples // n_subjects):
            # Randomly assign stress level
            stress_level = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
            
            # Generate physiological features based on stress level
            if stress_level == 0:  # Low stress
                hr = np.random.normal(baseline_hr, 5)
                hrv_rmssd = np.random.normal(baseline_hrv, 8)
                hrv_sdnn = np.random.normal(baseline_hrv * 1.2, 10)
                movement = np.random.normal(baseline_movement, 0.1)
                
            elif stress_level == 1:  # Medium stress
                hr = np.random.normal(baseline_hr + 15, 7)  # Increased HR
                hrv_rmssd = np.random.normal(baseline_hrv * 0.7, 6)  # Decreased HRV
                hrv_sdnn = np.random.normal(baseline_hrv * 0.8, 8)
                movement = np.random.normal(baseline_movement + 0.2, 0.15)  # More movement
                
            else:  # High stress
                hr = np.random.normal(baseline_hr + 25, 10)  # Much higher HR
                hrv_rmssd = np.random.normal(baseline_hrv * 0.5, 5)  # Much lower HRV
                hrv_sdnn = np.random.normal(baseline_hrv * 0.6, 6)
                movement = np.random.normal(baseline_movement + 0.4, 0.2)  # Much more movement
            
            # Add some noise and ensure realistic ranges
            hr = max(50, min(120, hr))  # Realistic HR range
            hrv_rmssd = max(10, hrv_rmssd)  # Minimum HRV
            hrv_sdnn = max(15, hrv_sdnn)  # Minimum HRV
            movement = max(0, movement)  # Non-negative movement
            
            # Additional derived features
            hr_variability = np.random.normal(hr * 0.1, 2)  # HR variability
            stress_index = (hr / hrv_rmssd) if hrv_rmssd > 0 else 0  # Stress index
            
            data.append({
                'subject_id': subject_id,
                'heart_rate': hr,
                'hrv_rmssd': hrv_rmssd,
                'hrv_sdnn': hrv_sdnn,
                'movement_intensity': movement,
                'hr_variability': hr_variability,
                'stress_index': stress_index,
                'stress_level': stress_level
            })
    
    return pd.DataFrame(data)

# Generate the dataset
print("Generating synthetic stress dataset...")
df = generate_synthetic_stress_data(n_samples=10000, n_subjects=50)
print(f"Dataset created: {df.shape}")
print(f"\nStress level distribution:")
print(df['stress_level'].value_counts().sort_index())
print(f"\nFirst few rows:")
print(df.head())

# Data overview
print("\n=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"\nFeature statistics:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nStress level distribution:")
stress_counts = df['stress_level'].value_counts().sort_index()
for level, count in stress_counts.items():
    percentage = count / len(df) * 100
    print(f"  Level {level}: {count} samples ({percentage:.1f}%)")

# Visualize the data distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

features = ['heart_rate', 'hrv_rmssd', 'hrv_sdnn', 'movement_intensity', 'hr_variability', 'stress_index']
feature_names = ['Heart Rate (bpm)', 'HRV RMSSD (ms)', 'HRV SDNN (ms)', 'Movement Intensity', 'HR Variability', 'Stress Index']

for i, (feature, name) in enumerate(zip(features, feature_names)):
    for stress_level in [0, 1, 2]:
        data = df[df['stress_level'] == stress_level][feature]
        axes[i].hist(data, alpha=0.6, label=f'Stress Level {stress_level}', bins=30)
    
    axes[i].set_title(f'{name} by Stress Level')
    axes[i].set_xlabel(name)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.drop('subject_id', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Prepare features and target
feature_columns = ['heart_rate', 'hrv_rmssd', 'hrv_sdnn', 'movement_intensity', 'hr_variability', 'stress_index']
X = df[feature_columns]
y = df['stress_level']

print(f"\nFeatures: {feature_columns}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeatures scaled successfully")

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

print("Model trained successfully!")
print(f"\nTraining accuracy: {rf_model.score(X_train_scaled, y_train):.3f}")
print(f"Test accuracy: {rf_model.score(X_test_scaled, y_test):.3f}")

# Classification report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Low Stress', 'Medium Stress', 'High Stress']))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Stress Level')
plt.ylabel('Actual Stress Level')
plt.show()

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance in Stress Detection')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# SHAP values for explainability
print("\nComputing SHAP values for model explainability...")

# Use a subset for SHAP (it can be computationally expensive)
X_test_sample = X_test_scaled[:1000]  # Use first 1000 test samples

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_sample)

# Plot SHAP summary
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_columns, show=False)
plt.title('SHAP Summary Plot - Feature Impact on Stress Prediction')
plt.tight_layout()
plt.show()

print("SHAP analysis completed!")

def predict_stress_level(heart_rate, hrv_rmssd, hrv_sdnn, movement_intensity, hr_variability, stress_index):
    """
    Predict stress level from physiological features
    """
    # Create input array
    input_data = np.array([[heart_rate, hrv_rmssd, hrv_sdnn, movement_intensity, hr_variability, stress_index]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = rf_model.predict(input_scaled)[0]
    probabilities = rf_model.predict_proba(input_scaled)[0]
    
    return prediction, probabilities

# Example predictions
print("\n=== EXAMPLE PREDICTIONS ===")

# Example 1: Low stress scenario
print("\n1. Low Stress Scenario:")
print("   Heart Rate: 70 bpm, HRV RMSSD: 50 ms, Movement: 0.3")
pred, prob = predict_stress_level(70, 50, 60, 0.3, 7, 1.4)
print(f"   Predicted Stress Level: {pred}")
print(f"   Probabilities: Low={prob[0]:.3f}, Medium={prob[1]:.3f}, High={prob[2]:.3f}")

# Example 2: High stress scenario
print("\n2. High Stress Scenario:")
print("   Heart Rate: 95 bpm, HRV RMSSD: 25 ms, Movement: 0.8")
pred, prob = predict_stress_level(95, 25, 30, 0.8, 12, 3.8)
print(f"   Predicted Stress Level: {pred}")
print(f"   Probabilities: Low={prob[0]:.3f}, Medium={prob[1]:.3f}, High={prob[2]:.3f}")

# Example 3: Medium stress scenario
print("\n3. Medium Stress Scenario:")
print("   Heart Rate: 85 bpm, HRV RMSSD: 35 ms, Movement: 0.6")
pred, prob = predict_stress_level(85, 35, 42, 0.6, 9, 2.4)
print(f"   Predicted Stress Level: {pred}")
print(f"   Probabilities: Low={prob[0]:.3f}, Medium={prob[1]:.3f}, High={prob[2]:.3f}")

# Analyze what the model learned
print("\n=== MODEL INSIGHTS ===")
print("\n1. Most Important Features for Stress Detection:")
for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
    print(f"   {i}. {row['feature']}: {row['importance']:.3f}")

print("\n2. Physiological Patterns by Stress Level:")
stress_patterns = df.groupby('stress_level')[feature_columns].mean()
print(stress_patterns.round(2))

print("\n3. Key Insights:")
print("   • Heart rate increases with stress level")
print("   • HRV (both RMSSD and SDNN) decreases with stress")
print("   • Movement intensity increases with stress")
print("   • Stress index (HR/HRV ratio) is a strong predictor")

print("\n4. Model Performance:")
print(f"   • Overall accuracy: {rf_model.score(X_test_scaled, y_test):.1%}")
print(f"   • Good at distinguishing between stress levels")
print(f"   • Most confident predictions for extreme stress levels")

# Save the model and scaler
import joblib

# Save model
joblib.dump(rf_model, 'stress_detection_model.pkl')
joblib.dump(scaler, 'stress_detection_scaler.pkl')

print("\nModel and scaler saved successfully!")
print("Files created:")
print("  - stress_detection_model.pkl")
print("  - stress_detection_scaler.pkl")

# Create a simple usage example
usage_example = """
# Load the model
import joblib
import numpy as np

model = joblib.load('stress_detection_model.pkl')
scaler = joblib.load('stress_detection_scaler.pkl')

# Predict stress level
def predict_stress(heart_rate, hrv_rmssd, hrv_sdnn, movement_intensity, hr_variability, stress_index):
    input_data = np.array([[heart_rate, hrv_rmssd, hrv_sdnn, movement_intensity, hr_variability, stress_index]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    return prediction, probabilities

# Example usage
pred, prob = predict_stress(85, 35, 42, 0.6, 9, 2.4)
print(f"Predicted stress level: {pred}")
print(f"Probabilities: {prob}")
"""

with open('usage_example.py', 'w') as f:
    f.write(usage_example)

print("\nUsage example saved to 'usage_example.py'")

