import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json

print("=" * 60)
print("Medical AI Model Training Script")
print("=" * 60)

# Get the project root directory (parent of training directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 1. Load the dataset
print("\n[1/6] Loading dataset...")
dataset_path = os.path.join(project_root, 'datasets', 'CVD_cleaned.csv')
df = pd.read_csv(dataset_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Define features and targets
print("\n[2/6] Preparing features and targets...")
target_columns = ['Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 
                  'Depression', 'Diabetes', 'Arthritis']
feature_columns = [col for col in df.columns if col not in target_columns]

# Separate categorical and numerical features
categorical_features = ['General_Health', 'Checkup', 'Exercise', 'Sex', 
                       'Age_Category', 'Smoking_History']
numerical_features = ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 
                     'Fruit_Consumption', 'Green_Vegetables_Consumption', 
                     'FriedPotato_Consumption']

X = df[feature_columns]
y = df[target_columns]

print(f"Features: {len(feature_columns)}")
print(f"Targets: {len(target_columns)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")

# 3. Create and fit preprocessor
print("\n[3/6] Creating and fitting preprocessor...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop=None, sparse_output=False), categorical_features)
    ]
)

# Fit preprocessor on all data to learn all categories
X_processed = preprocessor.fit_transform(X)

# Pad to 36 features if needed
if X_processed.shape[1] < 36:
    padding_size = 36 - X_processed.shape[1]
    padding = np.zeros((X_processed.shape[0], padding_size))
    X_processed = np.hstack([X_processed, padding])
    print(f"Padded to {X_processed.shape[1]} features")

print(f"Processed features shape: {X_processed.shape}")

# 4. Split data
print("\n[4/6] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=None
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 5. Train model
print("\n[5/6] Training MultiOutputClassifier with RandomForestClassifier...")
# RandomForest handles class imbalance better than LogisticRegression
# Use class_weight='balanced' to handle class imbalance
# Increased n_estimators and removed max_depth limit for better learning
base_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,  # No limit - let trees grow fully
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model = MultiOutputClassifier(base_classifier, n_jobs=-1)

model.fit(X_train, y_train)
print("Model training completed!")

# 6. Evaluate model
print("\n[6/6] Evaluating model performance...")
y_pred = model.predict(X_test)

# Calculate overall accuracy (average of all diseases)
disease_accuracies = {}
total_accuracy = 0
for i, disease in enumerate(target_columns):
    disease_accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    disease_accuracies[disease] = float(disease_accuracy)
    total_accuracy += disease_accuracy

overall_accuracy = total_accuracy / len(target_columns)

print(f"\n{'='*60}")
print(f"Overall Accuracy (Average): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"{'='*60}")

# Per-disease accuracy
print("\nPer-Disease Accuracy:")
print("-" * 60)
for i, disease in enumerate(target_columns):
    print(f"{disease:20s}: {disease_accuracies[disease]:.4f} ({disease_accuracies[disease]*100:.2f}%)")

# Detailed classification report
print("\n" + "="*60)
print("Detailed Classification Report:")
print("="*60)
for i, disease in enumerate(target_columns):
    print(f"\n{disease}:")
    unique_classes = sorted(set(y_test.iloc[:, i].unique()) | set(y_pred[:, i]))
    print(classification_report(y_test.iloc[:, i], y_pred[:, i], 
                                zero_division=0))

# 7. Save model and preprocessor
print("\n" + "="*60)
print("Saving model and preprocessor...")
print("="*60)

# Create models directory if it doesn't exist
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

# Save model
model_path = os.path.join(models_dir, 'medical_ai_model.pkl')
joblib.dump(model, model_path)
print(f"[OK] Model saved to: {model_path}")

# Save preprocessor
preprocessor_path = os.path.join(models_dir, 'preprocessor_36.pkl')
joblib.dump(preprocessor, preprocessor_path)
print(f"[OK] Preprocessor saved to: {preprocessor_path}")

# 8. Save training metadata
metadata = {
    'model_type': 'MultiOutputClassifier with RandomForestClassifier',
    'n_features': int(X_processed.shape[1]),
    'n_targets': len(target_columns),
    'target_columns': target_columns,
    'feature_columns': feature_columns,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'train_size': int(X_train.shape[0]),
    'test_size': int(X_test.shape[0]),
    'overall_accuracy': float(overall_accuracy),
    'disease_accuracies': disease_accuracies,
    'random_state': 42,
    'test_size_ratio': 0.2
}

metadata_path = os.path.join(models_dir, 'training_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Training metadata saved to: {metadata_path}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nModel files saved in 'models/' directory:")
print(f"  - medical_ai_model.pkl")
print(f"  - preprocessor_36.pkl")
print(f"  - training_metadata.json")
print(f"\nYou can now use these files in your Flask app!")

