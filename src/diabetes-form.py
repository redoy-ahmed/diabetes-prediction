import sys
import pandas as pd
import joblib
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QLabel, \
    QHBoxLayout


class DiabetesForm(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Diabetes Prediction Form')

        self.loadModel()
        self.createUI()

    def loadModel(self):
        try:
            self.df = joblib.load('diabetes_model.pkl')
            print("DataFrame loaded successfully from 'diabetes_model.pkl'.")
        except FileNotFoundError:
            print("Error: The file 'diabetes_model.pkl' was not found.")
            raise

    def createUI(self):
        self.submit_button = QPushButton('Submit', self)

        # Set button width smaller
        self.submit_button.setFixedHeight(60)
        self.submit_button.setFixedWidth(150)
        self.submit_button.setStyleSheet("font-size: 20px;")
        self.submit_button.clicked.connect(self.submitClicked)

        self.table_widget = QTableWidget(self)
        self.loadRandomData()

        # Create a QLabel object
        self.label = QLabel("Please select a row from the Grid")

        # Create a message label to display prediction outcome
        self.prediction_label = QLabel('Prediction:')
        self.prediction_label.setStyleSheet("font-size: 20px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addWidget(self.label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.submit_button)

        self.adjustWindowSize()

    def loadRandomData(self):
        try:
            self.df_csv = pd.read_csv('diabetes.csv')
            random_rows = self.df_csv.sample(n=10)
            self.populateGrid(random_rows)
        except FileNotFoundError:
            print("Error: The file 'diabetes.csv' was not found.")
            raise

    def populateGrid(self, data):
        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(data.columns))
        self.table_widget.setHorizontalHeaderLabels(data.columns)

        # Make header row bold
        header_font = QFont()
        header_font.setBold(True)
        self.table_widget.horizontalHeader().setFont(header_font)

        for i, row in enumerate(data.values):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.table_widget.setItem(i, j, item)

        # Connect item click signal to the slot
        self.table_widget.itemClicked.connect(self.on_table_item_click)

    def adjustWindowSize(self):
        table_width = self.table_widget.verticalHeader().width() + self.table_widget.horizontalHeader().length() + 60
        table_height = self.table_widget.verticalHeader().length() + self.table_widget.horizontalHeader().height()
        self.resize(table_width, table_height)

    def on_table_item_click(self, item):
        # Select the entire row of the clicked item
        row = item.row()
        self.table_widget.selectRow(row)
        self.label.setVisible(False)

    def submitClicked(self):
        # Check if any row is selected
        if self.table_widget.selectedItems():
            self.testSelectedRow()
        else:
            self.showMessage("Please select a row from the Grid")

    def testSelectedRow(self):
        self.selected_row = self.table_widget.currentRow()

        self.row_data = [self.table_widget.item(self.selected_row, column).text()
                         for column in range(self.table_widget.columnCount() - 1)]

        try:
            self.row_data_numeric = [float(value) for value in self.row_data]

            if len(self.df_csv.columns) > 1:
                column_names = self.df_csv.columns[:-1]
                self.row_data_named = pd.DataFrame([self.row_data_numeric], columns=column_names)
                self.predictOutcome()
        except ValueError:
            print("Error: Invalid input data.")
        except Exception as e:
            print("Error occurred during prediction:", e)

    def predictOutcome(self):
        prediction = int(self.df.predict(self.row_data_named)[0])
        print("Predicted outcome:", prediction)
        self.table_widget.clearSelection()

        if prediction == 1:
            self.prediction_label.setText("Prediction: Diabetic")
            self.prediction_label.setStyleSheet("color: red; font-size: 20px;")
        else:
            self.prediction_label.setText("Prediction: Non-Diabetic")
            self.prediction_label.setStyleSheet("color: green; font-size: 20px;")

    def showMessage(self, msg):
        self.label.setText(msg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DiabetesForm()
    window.show()
    sys.exit(app.exec_())
