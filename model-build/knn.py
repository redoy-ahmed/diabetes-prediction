import sys

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
try:
    # Determine the environment
    if sys.platform == "darwin":
        data = pd.read_csv('/Users/redoyahmed/Desktop/Python Projects/diabetes-prediction/data/diabetes.csv')
    else:
        data = pd.read_csv('../data/diabetes.csv')
    
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The file 'diabetes.csv' was not found.")
    raise

# Specify feature columns and target column
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                   'DiabetesPedigreeFunction', 'Age']
target_column = 'Outcome'

# Prepare features (X) and target (y)
X = data[feature_columns]
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the k-NN model
k = 10  # You can adjust the number of neighbors
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Test on a single data point
single_data = X_test.iloc[0].values.reshape(1, -1)
single_data_df = pd.DataFrame(single_data, columns=feature_columns)  # Ensure proper feature names
single_pred = model.predict(single_data_df)
single_prob = model.predict_proba(single_data_df)

# Map prediction to "Yes" or "No"
prediction_label = "Yes" if single_pred[0] == 1 else "No"

print(f"Diabetes: {prediction_label}")
print(f"Prediction probabilities No and Yes: {single_prob[0]}")

# Determine the environment
if sys.platform == "darwin":
    joblib.dump(model, '/Users/redoyahmed/Desktop/Python Projects/diabetes-prediction/models/diabetes_model_knn.pkl')
else:
    joblib.dump(model, '../models/diabetes_model_knn.pkl')

print("Model saved as 'diabetes_model_knn.pkl'.")
