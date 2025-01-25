import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# Load your dataset
data_path = 'arabica_data_cleaned.csv'
data = pd.read_csv(data_path)

# Select only the relevant features and target
features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Moisture']
target = 'Total.Cup.Points'

# Drop any rows with missing values
data = data.dropna(subset=features + [target])

# Split into input (X) and output (y)
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalize target (y) to [0, 1] range
y_min, y_max = 50, 100
y_train_scaled = (y_train - y_min) / (y_max - y_min)
y_test_scaled = (y_test - y_min) / (y_max - y_min)

# Load the neural network model
nn_model = tf.keras.models.load_model('coffee_model.keras')

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions using both models
nn_predictions = nn_model.predict(X_test_scaled) * (y_max - y_min) + y_min
rf_predictions = rf_model.predict(X_test_scaled)

# Calculate MAE, MSE, and R² for both models
nn_mae = mean_absolute_error(y_test, nn_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
nn_mse = mean_squared_error(y_test, nn_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
nn_r2 = r2_score(y_test, nn_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("Neural Network MAE:", nn_mae)
print("Random Forest MAE:", rf_mae)
print("Neural Network MSE:", nn_mse)
print("Random Forest MSE:", rf_mse)
print("Neural Network R²:", nn_r2)
print("Random Forest R²:", rf_r2)

# Plot actual vs predicted values for both models
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, nn_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Neural Network: Actual vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest: Actual vs Predicted")

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# --- Feature Importance for Random Forest ---
feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# --- Save the Random Forest model and scaler ---
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

with open('rf_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Random Forest model and scaler saved.")

# --- Neural Network Model Parameters (Weights and Biases) ---
print("\n--- Neural Network Model Parameters (Weights and Biases) ---\n")
for i, layer in enumerate(nn_model.layers):
    weights, biases = layer.get_weights()
    print(f"Layer {i + 1} - {layer.name}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights: \n{weights}\n")
    print(f"Biases shape: {biases.shape}")
    print(f"Biases: \n{biases}\n")

# --- Random Forest Model Parameters ---
print("\n--- Random Forest Model Parameters ---\n")
print(f"Number of Estimators: {rf_model.n_estimators}")
print(f"Max Features: {rf_model.max_features}")
print(f"Random State: {rf_model.random_state}")

# Example prediction for new data
example_input = [[8, 8.58, 8.17, 8.17, 8, 0]]
example_input_scaled = scaler.transform(example_input)

nn_example_pred = nn_model.predict(example_input_scaled) * (y_max - y_min) + y_min
rf_example_pred = rf_model.predict(example_input_scaled)

print(f"\nNeural Network Prediction for example input: {nn_example_pred[0][0]}")
print(f"Random Forest Prediction for example input: {rf_example_pred[0]}")
