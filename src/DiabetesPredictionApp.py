import sys

import joblib
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, \
    QGridLayout, QLineEdit, QMessageBox
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DiabetesPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.fontSize = '20px'
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

        self.initUI()

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

    def initUI(self):
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
        grid_layout.addWidget(self.addQLabel("Pregnancies"), 0, 0)
        self.pregnancyInput = QLineEdit()
        self.pregnancyInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.pregnancyInput.setValidator(QIntValidator(0, 20))
        grid_layout.addWidget(self.pregnancyInput, 0, 1)

        grid_layout.addWidget(self.addQLabel("Glucose level:"), 1, 0)
        self.glucoseInput = QLineEdit()
        self.glucoseInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.glucoseInput.setValidator(QIntValidator(0, 300))
        grid_layout.addWidget(self.glucoseInput, 1, 1)

        grid_layout.addWidget(self.addQLabel("Blood Pressure:"), 2, 0)
        self.bloodPressureInput = QLineEdit()
        self.bloodPressureInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.bloodPressureInput.setValidator(QIntValidator(0, 200))
        grid_layout.addWidget(self.bloodPressureInput, 2, 1)

        grid_layout.addWidget(self.addQLabel("Skin Thickness:"), 3, 0)
        self.skinThicknessInput = QLineEdit()
        self.skinThicknessInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.skinThicknessInput.setValidator(QIntValidator(0, 100))
        grid_layout.addWidget(self.skinThicknessInput, 3, 1)

        grid_layout.addWidget(self.addQLabel("Insulin Level:"), 4, 0)
        self.insulinInput = QLineEdit()
        self.insulinInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.insulinInput.setValidator(QIntValidator(0, 900))
        grid_layout.addWidget(self.insulinInput, 4, 1)

        grid_layout.addWidget(self.addQLabel("Body Mass Index:"), 5, 0)
        self.bmiInput = QLineEdit()
        self.bmiInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.bmiInput.setValidator(QDoubleValidator(0, 100, 2))
        grid_layout.addWidget(self.bmiInput, 5, 1)

        grid_layout.addWidget(self.addQLabel("Diabetes Pedigree Function:"), 6, 0)
        self.dpfInput = QLineEdit()
        self.dpfInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.dpfInput.setValidator(QDoubleValidator(0, 3, 2))
        grid_layout.addWidget(self.dpfInput, 6, 1)

        grid_layout.addWidget(self.addQLabel("Age:"), 7, 0)
        self.ageInput = QLineEdit()
        self.ageInput.setStyleSheet(f'font-size: {self.fontSize}')
        self.ageInput.setValidator(QIntValidator(0, 120))
        grid_layout.addWidget(self.ageInput, 7, 1)

        main_layout.addLayout(grid_layout)

        # Result label
        self.resultLabel = QLabel('Result')
        self.resultLabel.setFont(QFont('Arial', 16))
        self.resultLabel.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.resultLabel)

        # Predict button
        self.predictButton = QPushButton('Predict')
        self.predictButton.setFont(QFont('Arial', 14))
        self.predictButton.clicked.connect(self.predictDiabetes)
        main_layout.addWidget(self.predictButton)

        # Load Random Data button
        self.loadDataButton = QPushButton('Load Random Data')
        self.loadDataButton.setFont(QFont('Arial', 14))
        self.loadDataButton.clicked.connect(self.loadRandomData)
        main_layout.addWidget(self.loadDataButton)

        # Set main layout
        self.setLayout(main_layout)

    def addQLabel(self, text):
        qLabel = QLabel(text)
        qLabel.setStyleSheet(f'font-size: {self.fontSize}')
        return qLabel

    def loadRandomData(self):
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

    def predictDiabetes(self):
        # Check if all inputs are filled
        inputs = [
            self.pregnancyInput, self.glucoseInput, self.bloodPressureInput,
            self.skinThicknessInput, self.insulinInput, self.bmiInput,
            self.dpfInput, self.ageInput
        ]
        for input_field in inputs:
            if not input_field.text():
                QMessageBox.warning(self, "Input Error", "Please fill all input fields.")
                return

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
