import time
import pickle
import lightgbm
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# Streamlit Frontend
st.markdown("<h2 style='text-align: center;'><u>Santander Customer Transaction Prediction</u></h2>", unsafe_allow_html=True)
st.write("Santander Bank, N. A., formerly Sovereign Bank, is a wholly owned subsidiary of the Spanish Santander Group. It is based in Boston and its principal market is the northeastern United States. Sovereign Bank was rebranded as Santander Bank on October 17, 2013. In 2018, Santander bank hosted a competition on kaggle to predict whether a particular customer with given details would transact or not. Below are the scores:")
col1, col2, col3, col4, col5 = st.beta_columns([2.9, 1.9, 1.65, 2.4, 2])

with col1:
	st.success("Kaggle score: 90%")
with col2:
	st.success("Rank: 374")
with col3:
	st.success("Top 4%")
with col4:
	st.success("Accuracy 98%")
with col5:
	st.success("F1 90%")

st.write("Please provide a text file with all the features in order to predict which customer will make a specific transaction in the future, irrespective of the amount of money transacted.")


# Load file
uploaded_file = st.file_uploader("",type='txt')

# Light GBM Model
def run_model(uploaded_file):
	"""
	Main function to run ML model
	"""
	try:
		start_time = time.time()
		model = pickle.load(open("finalized_model.pkl", 'rb'))
		file = pd.read_csv(uploaded_file, sep='\t')
		autoencoder = load_model("autoencoder_model.h5")
		extra_features = autoencoder.predict(file.iloc[0, 1:].values.astype('float32').reshape(1, 200))
		total_features = np.append(file.iloc[0, 1:].values.astype('float32').reshape(1, 200), extra_features).reshape(1, -1)
		if model.predict(total_features)[0] ==1:
			st.success('Transacted.')
			st.info(f'Execution time: {round(time.time() - start_time, 2)} seconds')
		else:
			st.success('No Transaction.')
	except ValueError:
		st.warning('Upload a text file.')


# Run Model
if st.button('Predict'):
	run_model(uploaded_file)
	

# streamlit run CS1_Frontend.py

