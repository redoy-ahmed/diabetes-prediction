# Diabetes Prediction Application using ML Models

![Diabetes Prediction](https://github.com/redoy-ahmed/diabetes-prediction/blob/master/img2.png)

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
   python DiabetesPredictionApp.py
   ```

3. The application window will open. You can input your medical information into the fields provided.

4. Click on the "Predict" button to get the prediction result.

5. You can also click on the "Load Random Data" button to load a random sample from the dataset for testing.

## Code Overview

### Main Application

The main application is defined in `app.py`. It includes:

- **Model Loading**: The `loadModel` method loads the kNN model and the dataset.
- **UI Initialization**: The `initUI` method sets up the user interface including input fields, labels, and buttons.
- **Event Handling**: The `loadRandomData` and `predictDiabetes` methods handle the button click events to load data and make predictions, respectively.

### Helper Functions

- **addQLabel**: Creates and returns a `QLabel` with the specified text and styles.
- **addQLineEdit**: Creates and returns a `QLineEdit` with a validator for user input.

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


   
3. Interact with the GUI:

- The application will display a table with random samples from the diabetes.csv file.
- Select a row from the table to test the data.
- Click the "Submit" button to see the predictions from both the k-NN and Naive Bayes models.

## Acknowledgments
The dataset used for training the models is sourced from the PIMA Indians Diabetes Database.
Thanks to the developers of the Python packages used in this project: Pandas, scikit-learn, joblib, and PyQt5.