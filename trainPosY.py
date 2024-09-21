import pandas as pd
from pycaret.regression import *
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('rangemap_1000_rows.csv')

# Select only the 'x' column and the target 'pos_y'
data = data[['x', 'y', 'pos_y']]

# Initialize PyCaret setup for regression
exp1 = setup(data, target='pos_y', session_id=123)

# Compare machine learning models
best_model = compare_models()

# Create and tune the best model
tuned_model = tune_model(best_model)

# Save the model locally
save_model(tuned_model, 'pos_y_model')

# Calculate Mean Squared Error on validation data
# The 'get_config' function retrieves the current dataset
X_train = get_config('X_train')
y_train = get_config('y_train')
X_test = get_config('X_test')
y_test = get_config('y_test')

# Make predictions on the test set
predictions = predict_model(tuned_model, data=X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions['prediction_label'])

print(f'Mean Squared Error: {mse:.2f}')