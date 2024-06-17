import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
dataset = pd.read_csv('brain_stroke.csv')

# Print class distribution
print("Class distribution in dataset:")
print(dataset['stroke'].value_counts())

# Prepare features and target
X = dataset.drop('stroke', axis=1)
y = dataset['stroke']

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Create preprocessing pipelines for both numeric and categorical data
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Create a pipeline that first preprocesses the data and then trains the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced'))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Define input data for prediction
input_data = {
    'gender': ['Male'],
    'age': [80],
    'hypertension': [0],
    'heart_disease': [1],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [76.6],
    'bmi': [36.6],
    'smoking_status': ['formerly smoked']
}

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

# Debugging: Check the preprocessing output
print("Input Data Before Preprocessing:\n", input_df)

input_data_transformed = model_pipeline.named_steps['preprocessor'].transform(input_df)
print("Input Data After Preprocessing:\n", input_data_transformed)

# Make predictions
prediction = model_pipeline.predict(input_df)
print("Prediction:", prediction)

# Save the model
filename = 'brain_stroke_model.pkl'
joblib.dump(model_pipeline, filename)
print(f"Model saved to {filename}")
