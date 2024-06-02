# Diabetes Prediction Application

## Overview

This application uses k-Nearest Neighbors (k-NN) and Naive Bayes models to predict whether a person has diabetes based on various health parameters. The user can interact with the application via a graphical user interface (GUI) built using PyQt5.

## Project Files

- `diabetes.csv`: The dataset containing health parameters and diabetes outcomes.
- `diabetes_model_knn.pkl`: The trained k-Nearest Neighbors model.
- `diabetes_model_nb.pkl`: The trained Naive Bayes model.
- `diabetes_prediction.py`: The main Python script that loads the models, reads the data, and runs the PyQt5 GUI for making predictions.

## Installation

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Required Python Packages

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### requirements.txt should contain:

- pandas
- joblib
- scikit-learn
- PyQt5

## Model Training
If you need to retrain the models, use the following scripts:

Train k-Nearest Neighbors Model

```bash
python knn.py
```

Train gausian Naive Bayes Model

```bash
python gausian_naive_bayes.py
```

## Usage
1. Ensure the dataset and models are available:

      Place the following files in the same directory as diabetes_prediction.py:

      - diabetes.csv
      - diabetes_model_knn.pkl
      - diabetes_model_nb.pkl
2. Run the application:

      Execute the main script using Python:

   ```bash
   python diabetes_prediction.py
   ```
   
3. Interact with the GUI:

- The application will display a table with random samples from the diabetes.csv file.
- Select a row from the table to test the data.
- Click the "Submit" button to see the predictions from both the k-NN and Naive Bayes models.

## Acknowledgments
The dataset used for training the models is sourced from the PIMA Indians Diabetes Database.
Thanks to the developers of the Python packages used in this project: Pandas, scikit-learn, joblib, and PyQt5.