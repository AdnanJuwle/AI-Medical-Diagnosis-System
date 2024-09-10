import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('C:/Users/adnan/AI-Medical-Diagnosis-System/datasets/survey_lung_cancer.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to disk
with open('C:/Users/adnan/AI-Medical-Diagnosis-System/models/lung_cancer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")
