import sys

import joblib
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, \
    QGridLayout, QLineEdit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DiabetesPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.df = None
        self.accuracy = None
        self.loadModel()
        self.pregnancyInput = None
        self.glucoseInput = None
        self.bloodPressureInput = None
        self.skinThicknessInput = None
        self.insulinInput = None
        self.bmiInput = None
        self.dpfInput = None
        self.ageInput = None
        self.resultLabel = None
        self.predictButton = None
        self.loadDataButton = None

        self.init_ui()

    # Load the kNN model and calculate its accuracy
    def loadModel(self):
        try:
            self.model = joblib.load('../models/diabetes_model_knn.pkl')
            print("Model loaded successfully.")

            # Load the dataset to calculate accuracy
            self.df = pd.read_csv('../data/diabetes.csv')
            featureColumns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                              'DiabetesPedigreeFunction', 'Age']
            targetColumn = 'Outcome'
            X = self.df[featureColumns]
            y = self.df[targetColumn]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Calculate accuracy for kNN
            y_pred_knn = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred_knn)
            print(f"kNN Model Accuracy: {self.accuracy:.2f}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    def init_ui(self):
        # Set window properties
        self.setWindowTitle('Diabetes Prediction')
        self.setGeometry(100, 100, 600, 600)
        # Main layout
        main_layout = QVBoxLayout()
        # Title and description
        title = QLabel('Diabetes Prediction')
        title.setFont(QFont('Arial', 16))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        # Create grid layout for input fields
        grid_layout = QGridLayout()
        # Create and add widgets to the grid layout
        grid_layout.addWidget(QLabel('Number of Pregnancies:'), 0, 0)
        self.pregnancyInput = QLineEdit()
        self.pregnancyInput.setValidator(QIntValidator(0, 20))
        grid_layout.addWidget(self.pregnancyInput, 0, 1)

        grid_layout.addWidget(QLabel('Glucose Level:'), 1, 0)
        self.glucoseInput = QLineEdit()
        self.glucoseInput.setValidator(QIntValidator(0, 300))
        grid_layout.addWidget(self.glucoseInput, 1, 1)

        grid_layout.addWidget(QLabel('Blood Pressure:'), 2, 0)
        self.bloodPressureInput = QLineEdit()
        self.bloodPressureInput.setValidator(QIntValidator(0, 200))
        grid_layout.addWidget(self.bloodPressureInput, 2, 1)

        grid_layout.addWidget(QLabel('Skin Thickness:'), 3, 0)
        self.skinThicknessInput = QLineEdit()
        self.skinThicknessInput.setValidator(QIntValidator(0, 100))
        grid_layout.addWidget(self.skinThicknessInput, 3, 1)

        grid_layout.addWidget(QLabel('Insulin Level:'), 4, 0)
        self.insulinInput = QLineEdit()
        self.insulinInput.setValidator(QIntValidator(0, 900))
        grid_layout.addWidget(self.insulinInput, 4, 1)

        grid_layout.addWidget(QLabel('Body Mass Index:'), 5, 0)
        self.bmiInput = QLineEdit()
        self.bmiInput.setValidator(QDoubleValidator(0, 100, 2))
        grid_layout.addWidget(self.bmiInput, 5, 1)

        grid_layout.addWidget(QLabel('Diabetes Pedigree Function:'), 6, 0)
        self.dpfInput = QLineEdit()
        self.dpfInput.setValidator(QDoubleValidator(0, 3, 2))
        grid_layout.addWidget(self.dpfInput, 6, 1)

        grid_layout.addWidget(QLabel('Age:'), 7, 0)
        self.ageInput = QLineEdit()
        self.ageInput.setValidator(QIntValidator(0, 120))
        grid_layout.addWidget(self.ageInput, 7, 1)

        main_layout.addLayout(grid_layout)

        # Result label
        self.resultLabel = QLabel('Result')
        self.resultLabel.setFont(QFont('Arial', 14))
        self.resultLabel.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.resultLabel)

        # Predict button
        self.predictButton = QPushButton('Predict')
        self.predictButton.setFont(QFont('Arial', 12))
        self.predictButton.clicked.connect(self.predict_diabetes)
        main_layout.addWidget(self.predictButton)

        # Load Random Data button
        self.loadDataButton = QPushButton('Load Random Data')
        self.loadDataButton.setFont(QFont('Arial', 12))
        self.loadDataButton.clicked.connect(self.load_random_data)
        main_layout.addWidget(self.loadDataButton)

        # Set main layout
        self.setLayout(main_layout)

    def load_random_data(self):
        # Load a random sample from the dataset
        random_row = self.df.sample(n=1).iloc[0]
        self.pregnancyInput.setText(str(int(random_row['Pregnancies'])))
        self.glucoseInput.setText(str(random_row['Glucose']))
        self.bloodPressureInput.setText(str(random_row['BloodPressure']))
        self.skinThicknessInput.setText(str(random_row['SkinThickness']))
        self.insulinInput.setText(str(random_row['Insulin']))
        self.bmiInput.setText(str(random_row['BMI']))
        self.dpfInput.setText(str(random_row['DiabetesPedigreeFunction']))
        self.ageInput.setText(str(int(random_row['Age'])))

    def predict_diabetes(self):
        # Get values from input fields
        pregnancies = int(self.pregnancyInput.text())
        glucose = float(self.glucoseInput.text())
        blood_pressure = float(self.bloodPressureInput.text())
        skin_thickness = float(self.skinThicknessInput.text())
        insulin = float(self.insulinInput.text())
        bmi = float(self.bmiInput.text())
        dpf = float(self.dpfInput.text())
        age = int(self.ageInput.text())

        # Create a numpy array for prediction
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

        # Make prediction
        prediction = self.model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            self.resultLabel.setText(f"Result: Diabetic (Accuracy: {self.accuracy:.2f})")
            self.resultLabel.setStyleSheet("color: red; font-size: 20px;")
        else:
            self.resultLabel.setText(f"Result: Non-Diabetic (Accuracy: {self.accuracy:.2f})")
            self.resultLabel.setStyleSheet("color: green; font-size: 20px;")


# Create the application
app = QApplication(sys.argv)

# Create and show the form
form = DiabetesPredictionApp()
form.show()

# Run the application
sys.exit(app.exec_())
