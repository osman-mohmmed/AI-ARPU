# %%
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
from statsmodels.base.model import LikelihoodModel
from statsmodels.regression.linear_model import OLS
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
import statsmodels.multivariate.tests.results as path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.append(r'\code')
from LinearRegDiagnostic import LinearRegDiagnostic
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# Split index
split_index = int(len(df_filtered) * split_ratio)

# Split data
train_df = df_filtered[:split_index]
test_df = df_filtered[split_index:]

#
lable = df_filtered['arpu_9']
features = df_filtered.drop('arpu_9', axis=1)

train_df = pd.read_csv('.')

X_train = train_df[features.columns] 
X_test = test_df[features.columns] 
y_train = train_df['arpu_9']
y_test = test_df['arpu_9']

activation = train_df.iloc[7,:]

train_df.to_csv("../data/training_data.csv")
test_df.to_csv("../data/test_date.csv")
activation.to_csv("../data/activation_data.csv")

df_filtered.to_csv("../data/joint_data_collection.csv")

#OLS model

X = add_constant(X_train)
Y = y_train

ols_m = sm.OLS(Y,X).fit()

ols_m.save('../results/models/OLS_model.pickle')


print(ols_m.summary())

# %%
'''OLS Model visualization
'''

cls = LinearRegDiagnostic(ols_m)
vif, fig, ax = cls()
fig.savefig('../results/images/M_DiagnosticPlots.png')
print(vif)

# %%
''''Modeling:
        ANN'''

input_feature = X_train.values  
target_feature = y_train.values  
test_target = y_test.values
test_feature = X_test.values
 
# Defining the TensorFlow feedforward neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (X_train.shape[1],)),# Input layer
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    tf.keras.layers.Dense(1, activation= 'linear')  # Output layer with 1 neuron
])
 
# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',  # Mean squared error
              metrics=['mae'])
 
# Training the model
model.fit(X_train, target_feature, epochs=50, batch_size=32, verbose=1)
 
 #Save model
model.save('../results/models/ANN_model.keras')

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, test_target, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# %%
'''ANN Model Visualisation'''

# Generate predictions from the ANN model

y_pred = model.predict(X_test).flatten()  # Ensure predictions are 1D if needed
 
# Calculate best-fit line (y_pred as a function of y_test)

slope, intercept = np.polyfit(y_test, y_pred, 1)
 
# Create best-fit line

best_fit_line = slope * y_test + intercept
 
# Calculate performance metrics

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)
 
# Plot the results
plt.figure(figsize=(15, 6))


# plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Model Predictions')
plt.subplot(1,2,1)

# Plot actual values in one color (e.g., red)
plt.scatter(y_test, X_test['offnet_mou_6'], alpha=0.6, color='yellow', label='offnet minutes')
 # Plot predicted values in a different color (e.g., green)
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predicted Values')

plt.plot(y_test, best_fit_line, color='red', linewidth=2, label='Best Fit Line')
 
# Add annotations

plt.title('ANN Predictions vs Best Fit Line[one feature]', fontsize=14)

plt.xlabel('offnet minutes', fontsize=12)

plt.ylabel('Predicted Values (y_pred)', fontsize=12)

plt.legend(fontsize=10)
 
# Display metrics

plt.text(

    0.05, 0.95,

    f"MSE: {mse:.4f}\nR²: {r2:.4f}",

    fontsize=10,

    transform=plt.gca().transAxes,

    verticalalignment='top',

    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')

)
 
plt.grid(alpha=0.3)

#second plot
plt.subplot(1,2,2)
# Plot actual values

plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted Values',  )

plt.plot(y_test, best_fit_line, color='red', linewidth=2, label='Best Fit Line')
 

# Add annotations

plt.title('ANN Predictions vs Best Fit Line', fontsize=14)

plt.xlabel('True Values (y_test)', fontsize=12)

plt.ylabel('Predicted Values (y_pred)', fontsize=12)

plt.legend(fontsize=10)
 
# Display metrics

plt.text(

    0.05, 0.95,

    f"MSE: {mse:.4f}\nR²: {r2:.4f}",

    fontsize=10,

    transform=plt.gca().transAxes,

    verticalalignment='top',

    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')

)
 
plt.grid(alpha=0.3)

plt.legend(loc = "upper center")

plt.savefig('../results/images/M_best_fit_line_scatter.png')

plt.show()

 


