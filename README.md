# Online_fraud_detection
This Python script implements a fraud detection system using a Support Vector Machine (SVM) model from the scikit-learn library. The workflow is as follows:

Data Loading: The script begins by loading a dataset from a CSV file named transaction_data.csv, which contains transaction records that include various features relevant to fraud detection.

Feature and Target Selection: It defines the feature columns (amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, and newbalanceDest) used for prediction and the target column (isFraud) that indicates whether a transaction is fraudulent (1) or not (0).

Data Splitting: The dataset is split into training and testing sets using an 80-20 split (25% for testing) while preserving the proportion of fraud cases using stratify. This ensures that both training and testing datasets represent the original class distribution.

Feature Scaling: The features are standardized using StandardScaler to ensure that each feature contributes equally to the model’s performance. This is critical for algorithms like SVM that are sensitive to feature scales.

Model Training: An SVM classifier is instantiated with a linear kernel and balanced class weights to handle class imbalance in the dataset. The model is then trained on the scaled training data.

Prediction: The trained model is used to make predictions on the test set, identifying whether each transaction is fraudulent or not.

Model Evaluation: The script evaluates the model’s performance by calculating:

Confusion Matrix: A matrix that summarizes the performance of the classification model by comparing actual versus predicted values.
Accuracy Score: The proportion of correctly predicted instances out of the total instances in the test set.
Classification Report: A comprehensive report detailing precision, recall, F1-score, and support for each class, which provides deeper insights into the model's performance.
