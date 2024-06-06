import sys

import joblib
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, \
    QGridLayout, QLineEdit, QMessageBox


def addQPushButton(text, method):
    button = QPushButton(text)
    button.setFont(QFont('Arial', 14))
    button.setStyleSheet("border: 2px solid white; background-color: lightgray;")
    button.setFixedHeight(50)
    button.clicked.connect(method)
    return button


class DiabetesPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.fontSize = '20px'
        self.model = None
        self.df = None
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

    # Load the kNN model
    def loadModel(self):
        try:
            self.model = joblib.load('../models/diabetes_model_knn.pkl')
            print("Model loaded successfully.")

            # Load the dataset
            self.df = pd.read_csv('../data/diabetes.csv')

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    # Init UI components
    def initUI(self):
        # Set window properties
        self.setWindowTitle('Diabetes Prediction')
        self.setGeometry(100, 100, 600, 600)
        # Set background color
        self.setStyleSheet("background-color: lightblue;")
        # Main layout
        mainLayout = QVBoxLayout()
        # Title and description
        title = QLabel('Diabetes Prediction')
        title.setFont(QFont('Arial', 16))
        title.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(title)
        # Create grid layout for input fields
        gridLayout = QGridLayout()

        # Create and add widgets to the grid layout
        gridLayout.addWidget(self.addQLabel("Pregnancies"), 0, 0)
        self.pregnancyInput = self.addQLineEdit(QIntValidator(0, 20))
        gridLayout.addWidget(self.pregnancyInput, 0, 1)

        gridLayout.addWidget(self.addQLabel("Glucose level:"), 1, 0)
        self.glucoseInput = self.addQLineEdit(QIntValidator(0, 300))
        gridLayout.addWidget(self.glucoseInput, 1, 1)

        gridLayout.addWidget(self.addQLabel("Blood Pressure:"), 2, 0)
        self.bloodPressureInput = self.addQLineEdit(QIntValidator(0, 200))
        gridLayout.addWidget(self.bloodPressureInput, 2, 1)

        gridLayout.addWidget(self.addQLabel("Skin Thickness:"), 3, 0)
        self.skinThicknessInput = self.addQLineEdit(QIntValidator(0, 100))
        gridLayout.addWidget(self.skinThicknessInput, 3, 1)

        gridLayout.addWidget(self.addQLabel("Insulin Level:"), 4, 0)
        self.insulinInput = self.addQLineEdit(QIntValidator(0, 900))
        gridLayout.addWidget(self.insulinInput, 4, 1)

        gridLayout.addWidget(self.addQLabel("Body Mass Index:"), 5, 0)
        self.bmiInput = self.addQLineEdit(QDoubleValidator(0, 100, 2))
        gridLayout.addWidget(self.bmiInput, 5, 1)

        gridLayout.addWidget(self.addQLabel("Diabetes Pedigree Function:"), 6, 0)
        self.dpfInput = self.addQLineEdit(QDoubleValidator(0, 3, 3))
        gridLayout.addWidget(self.dpfInput, 6, 1)

        gridLayout.addWidget(self.addQLabel("Age:"), 7, 0)
        self.ageInput = self.addQLineEdit(QIntValidator(0, 120))
        gridLayout.addWidget(self.ageInput, 7, 1)
        mainLayout.addLayout(gridLayout)

        # Result label
        self.resultLabel = QLabel('Result')
        self.resultLabel.setFont(QFont('Arial', 16))
        self.resultLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(self.resultLabel)

        # Predict button
        self.predictButton = addQPushButton('Predict', self.predictDiabetes)
        mainLayout.addWidget(self.predictButton)

        # Load Random Data button
        self.loadDataButton = addQPushButton('Load Random Data', self.loadRandomData)
        mainLayout.addWidget(self.loadDataButton)

        # Set main layout
        self.setLayout(mainLayout)

    # Create QLabel using this function
    def addQLabel(self, text):
        qLabel = QLabel(text)
        qLabel.setStyleSheet(f'font-size: {self.fontSize}')
        return qLabel

    # Create QLineEdit using this function
    def addQLineEdit(self, validator):
        qLineEdit = QLineEdit()
        qLineEdit.setStyleSheet(
            f'font-size: {self.fontSize}; border: 2px solid white; background-color: lightgray;')
        qLineEdit.setValidator(validator)
        return qLineEdit

    # Load random data into form fields
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

    # Predict Diabetes from form input
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
            self.resultLabel.setText(f"Result: Diabetic")
            self.resultLabel.setFont(QFont('Arial', 16))
            self.resultLabel.setStyleSheet("color: red;")
        else:
            self.resultLabel.setText(f"Result: Non-Diabetic")
            self.resultLabel.setFont(QFont('Arial', 16))
            self.resultLabel.setStyleSheet("color: green;")


# Create the application
app = QApplication(sys.argv)

# Create and show the form
form = DiabetesPredictionApp()
form.show()

# Run the application
sys.exit(app.exec_())
