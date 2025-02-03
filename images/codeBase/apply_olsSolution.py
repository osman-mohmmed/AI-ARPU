import pandas as pd
import statsmodels.api as sm
from statsmodels.tools import add_constant

input_data = pd.read_csv('tmp/activationBase/activation_data.csv') 
lable = input_data['arpu_9']
features = input_data.drop('arpu_9', axis=1)
# features_reshaped = features.values.reshape(-1,1)

X = add_constant(input_data)
Y = lable

ols_m = sm.load('tmp/knowledgeBase/OLS_model.pickle')

prdiction = ols_m.predict(X)

print(f'OLS Model Predictions: {prdiction}')
print(f'Actual Activation Vlue: {lable.values}')