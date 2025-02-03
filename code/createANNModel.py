import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
from io import StringIO
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools import add_constant
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# Define the directory path
# learning_base_dir = r'./learningBase'

# Check if the directory exists, if not, create it
# if not os.path.exists(learning_base_dir):
#     os.makedirs(learning_base_dir)

# Define the split ratio
# split_ratio = 0.8

# Read data
# df_filtered = pd.read_csv("../data/joint_data_collection.csv")

# Split index
# split_index = int(len(df_filtered) * split_ratio)

# Split data
# train_df = df_filtered[:split_index]
# test_df = df_filtered[split_index:]

train_df = pd.read_csv('tmp/learningBase/train/training_data.csv')
test_df = pd.read_csv('tmp/learningBase/validation/test_data.csv')


# Define features and target
features = train_df.drop('arpu_9', axis=1)
lable = train_df['arpu_9']

X_train = train_df[features.columns]
X_test = test_df[features.columns]
y_train = train_df['arpu_9']
y_test = test_df['arpu_9']

# Define the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 1
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer 2
    tf.keras.layers.Dense(1, activation='linear')  # Output layer
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',  # Mean squared error loss
              metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save model
model.save('tmp/knowledgeBase/currentAiSolution.keras')

# Create a DataFrame from the training history
history_df = pd.DataFrame(history.history)
# Define the directory path again
learning_base_dir = 'tmp/learningBase'


# Store evaluation results in markdown
history_df.to_markdown(learning_base_dir + '/annEvaluation.md')  # Store evaluation results in markdown

# Total training iterations
num_epochs = len(history_df['loss'])
num_batches_per_epoch = len(X_train) // 32  # Assuming batch size is 32
num_iterations = num_epochs * num_batches_per_epoch

# Final loss and MAE
final_loss = history_df['loss'].iloc[-1]  # Last loss value
final_mae = history_df['mae'].iloc[-1]  # Last MAE value

# Print the results
print(f"Total Training Iterations: {num_iterations}")
print(f"Final Loss: {final_loss}")
print(f"Final MAE: {final_mae}")

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Generate predictions
y_pred = model.predict(X_test).flatten()  # Ensure predictions are 1D

# Calculate best-fit line
slope, intercept = np.polyfit(y_test, y_pred, 1)
best_fit_line = slope * y_test + intercept

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot training and testing curves
plt.figure(figsize=(15, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), history_df['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), history_df['mae'], label='Training MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training MAE over Epochs')
plt.legend()

# Save the training curves plot
plt.savefig(learning_base_dir + '/annTrainingCurves.png')

# Diagnostic Plot (Scatter Plot of Test vs Predicted)
plt.figure(figsize=(15, 6))

#Plot actual vs predicted values
# plt.subplot(1, 2, 1)
# plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted Values')
# plt.plot(y_test, best_fit_line, color='red', linewidth=2, label='Best Fit Line')
# plt.xlabel('True Values (y_test)')
# plt.ylabel('Predicted Values (y_pred)')
# plt.title('Test vs Predicted Values')
# plt.legend()

# Plot for one feature, e.g., 'offnet_mou_6'
plt.subplot(1, 2, 2)
plt.scatter(y_test, X_test['offnet_mou_6'], alpha=0.6, color='yellow', label='offnet minutes')
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predicted Values')
plt.plot(y_test, best_fit_line, color='red', linewidth=2, label='Best Fit Line')
plt.xlabel('True Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.title('Offnet Minutes vs Predicted Values')
plt.legend()

# Save the diagnostic plot
plt.savefig(learning_base_dir + '/annDiagnosticPlot.png')

# Scatter plot with R2 and MSE as annotation
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predicted Values')
plt.plot(y_test, best_fit_line, color='red', linewidth=2, label='Best Fit Line')
plt.title('Model Predictions vs Actual')
plt.xlabel('True Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.text(0.05, 0.95, f"MSE: {mse:.4f}\nR²: {r2:.4f}", fontsize=12, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
plt.legend()

# Save the scatter plot
plt.savefig(learning_base_dir + '/annScatterPlot.png')

# Store model performance in markdown format for documentation
metrics_summary = {
    'Total Training Iterations': num_iterations,
    'Final Loss': final_loss,
    'Final MAE': final_mae,
    'Test Loss': loss,
    'Test MAE': mae,
    'MSE': mse,
    'R²': r2
}

metrics_df = pd.DataFrame(metrics_summary, index=[0])
metrics_df.to_markdown(learning_base_dir + '/annMetricsSummary.md')

print("Training, evaluation, and visualization are saved successfully.")
