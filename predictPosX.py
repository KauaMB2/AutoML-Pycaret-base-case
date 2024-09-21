import pandas as pd
from pycaret.regression import load_model, predict_model

# Load the saved model
model = load_model('pos_x_model')

# Load the original dataset
data = pd.read_csv('rangemap_100_rows.csv')

# Select only the 'x' column
input_data = data[['x', 'y']]

# Generate predictions
predictions = predict_model(model, data=input_data)

# Print the predictions DataFrame to check its structure
print(predictions.head())

# Create a new DataFrame with the required columns
# Replace 'prediction' with the actual column name from the predictions DataFrame
result_df = pd.DataFrame({
    'x': input_data['x'],
    'pos_x': data['pos_x'],  # Keep the actual pos_x if available
    'predicted_pos_x': predictions.iloc[:, -1]  # Assuming the last column contains predicted values
})

# Save the new DataFrame to a CSV file
result_df.to_csv('predictionsPosX.csv', index=False)

print("Predictions saved to 'predictionsPosX.csv'")
