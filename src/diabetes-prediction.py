import sys
import pandas as pd
import joblib
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QLabel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DiabetesPrediction(QWidget):
    def __init__(self):
        super().__init__()
        self.df_knn = None
        self.df_nb = None
        self.df_csv = None
        self.selected_row = None
        self.row_data_named = None
        self.accuracy_knn = None
        self.accuracy_nb = None
        self.initUI()

    # Initialize the UI components
    def initUI(self):
        self.setWindowTitle('Diabetes Prediction Form')
        self.loadModels()
        self.createUI()

    # Load the kNN and Naive Bayes models and calculate their accuracies
    def loadModels(self):
        try:
            self.df_knn = joblib.load('../models/diabetes_model_knn.pkl')
            self.df_nb = joblib.load('../models/diabetes_model_nb.pkl')
            print("Models loaded successfully.")

            # Load the dataset to calculate accuracy
            df = pd.read_csv('../data/diabetes.csv')
            feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                               'DiabetesPedigreeFunction', 'Age']
            target_column = 'Outcome'
            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Calculate accuracy for kNN
            y_pred_knn = self.df_knn.predict(X_test)
            self.accuracy_knn = accuracy_score(y_test, y_pred_knn)

            # Calculate accuracy for Naive Bayes
            y_pred_nb = self.df_nb.predict(X_test)
            self.accuracy_nb = accuracy_score(y_test, y_pred_nb)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    # Create the UI elements and layout
    def createUI(self):
        self.submit_button = self.createButton('Submit', self.submitClicked, 150, 60, "font-size: 20px;")
        self.table_widget = QTableWidget(self)
        self.loadRandomData()
        self.label = QLabel("Please select a row from the Grid")
        self.labelkNN = "Prediction Using kNN Model:"
        self.labelnb = "Prediction Using Naive Bayes Model:"
        self.prediction_label_knn = QLabel(self.labelkNN)
        self.prediction_label_knn.setStyleSheet("font-size: 20px;")
        self.prediction_label_nb = QLabel(self.labelnb)
        self.prediction_label_nb.setStyleSheet("font-size: 20px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addWidget(self.label)
        layout.addWidget(self.prediction_label_knn)
        layout.addWidget(self.prediction_label_nb)
        layout.addWidget(self.submit_button)
        self.adjustWindowSize()

    # Helper function to create a button
    def createButton(self, text, func, width, height, style):
        button = QPushButton(text, self)
        button.setFixedSize(width, height)
        button.setStyleSheet(style)
        button.clicked.connect(func)
        return button

    # Load random data from the CSV file to display in the table
    def loadRandomData(self):
        try:
            self.df_csv = pd.read_csv('../data/diabetes.csv')
            random_rows = self.df_csv.sample(n=10)
            self.populateGrid(random_rows)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    # Populate the table with data
    def populateGrid(self, data):
        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(data.columns))
        self.table_widget.setHorizontalHeaderLabels(data.columns)
        self.setTableHeaderStyle()

        for i, row in enumerate(data.values):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.table_widget.setItem(i, j, item)

        self.table_widget.itemClicked.connect(self.on_table_item_click)

    # Set the style for the table headers
    def setTableHeaderStyle(self):
        header_style = "QHeaderView::section { border: 1px solid black; }"
        header_font = QFont()
        header_font.setBold(True)
        self.table_widget.horizontalHeader().setFont(header_font)
        self.table_widget.horizontalHeader().setStyleSheet(header_style)

    # Adjust the window size based on the table dimensions
    def adjustWindowSize(self):
        table_width = self.table_widget.verticalHeader().width() + self.table_widget.horizontalHeader().length() + 60
        table_height = self.table_widget.verticalHeader().length() + self.table_widget.horizontalHeader().height()
        self.resize(table_width, table_height)

    # Handle table item click event to select the entire row
    def on_table_item_click(self, item):
        self.table_widget.selectRow(item.row())
        self.label.setVisible(False)

    # Handle the submit button click event
    def submitClicked(self):
        if self.table_widget.selectedItems():
            self.testSelectedRow()
        else:
            self.showMessage("Please select a row from the Grid")

    # Test the selected row using the loaded models
    def testSelectedRow(self):
        try:
            self.selected_row = self.table_widget.currentRow()
            self.row_data_named = self.getRowData()
            self.predictOutcomeUsingKnn()
            self.predictOutcomeUsingNaiveBayes()
        except ValueError:
            print("Error: Invalid input data.")
        except Exception as e:
            print(f"Error occurred during prediction: {e}")

    # Get data from the selected row
    def getRowData(self):
        row_data = [float(self.table_widget.item(self.selected_row, column).text())
                    for column in range(self.table_widget.columnCount() - 1)]
        column_names = self.df_csv.columns[:-1]
        return pd.DataFrame([row_data], columns=column_names)

    # Generic function to predict the outcome and update the label
    def predictOutcome(self, model, label_text, to_be_shown, accuracy):
        prediction = int(model.predict(self.row_data_named)[0])
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        color = "red" if prediction == 1 else "green"
        to_be_shown.setText(f"{label_text} {result} (Accuracy: {accuracy:.2f})")
        to_be_shown.setStyleSheet(f"color: {color}; font-size: 20px;")

    # Predict the outcome using the kNN model
    def predictOutcomeUsingKnn(self):
        self.predictOutcome(self.df_knn, self.labelkNN, self.prediction_label_knn, self.accuracy_knn)
        self.table_widget.clearSelection()

    # Predict the outcome using the Naive Bayes model
    def predictOutcomeUsingNaiveBayes(self):
        self.predictOutcome(self.df_nb, self.labelnb, self.prediction_label_nb, self.accuracy_nb)
        self.table_widget.clearSelection()

    # Display a message to the user
    def showMessage(self, msg):
        self.label.setText(msg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DiabetesPrediction()
    window.show()
    sys.exit(app.exec_())
