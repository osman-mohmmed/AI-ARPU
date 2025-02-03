import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('tmp/knowledgeBase/currentAiSolution.keras')
input_data = pd.read_csv('tmp/activationBase/activation_data.csv') 
lable = input_data['arpu_9']
features = input_data.drop('arpu_9', axis=1)



predictions = model.predict(features)
print(f'AI Prediction: {predictions}')
print(f'Actual Value: {lable.values}')