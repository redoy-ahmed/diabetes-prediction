import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
try:
    data = pd.read_csv('diabetes.csv')
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

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
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

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=feature_columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.show()

# Test on a single data point
single_data = X_test.iloc[0].values.reshape(1, -1)
single_pred = model.predict(single_data)
single_prob = model.predict_proba(single_data)

print(f"Single data point prediction: {single_pred[0]}")
print(f"Prediction probabilities: {single_prob[0]}")
