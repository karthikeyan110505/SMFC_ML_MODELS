Soil Microbial Fuel Cell (SMFC) Voltage Prediction using Machine Learning

Welcome! This repository contains my project for monitoring and predicting voltage output in a Soil Microbial Fuel Cell (SMFC). The voltage was recorded every 6 hours over time, and three different machine learning models were developed and compared to predict future voltage trends.

Project Overview
Objective:
To monitor voltage trends in a SMFC setup and predict future voltage outputs using machine learning techniques.

Data Collection:
Voltage readings taken every 6 hours.
Collected over a specific duration to observe natural voltage fluctuations.

Machine Learning Models Used:
Random Forest Regressor (RFR)
Artificial Neural Network (ANN)
Long Short-Term Memory (LSTM)

Comparison:
Models were compared based on performance metrics like R² Score and Mean Squared Error (MSE) to determine the most accurate predictor.

Technologies & Tools
Programming Language: Python

Libraries:
pandas, numpy - Data manipulation
scikit-learn - RFR & ANN
TensorFlow/Keras - LSTM
matplotlib, seaborn - Visualization

Platform: Jupyter Notebook

Repository Structure
├── data/
│   └── data_smfc_data  # Raw voltage readings (6-hour interval)
├── models/
│   ├── RFR.py
│   ├── ANN.py
│   ├── LSTM.py
├── notebooks/
│   ├── RFR.ipynb
│   ├── ANN.ipynb
│   ├── LSTM.ipynb
├── requirments/
│   ├──requirements.txt
├── README.md   # This file

How to Run
1️. Clone the repo
git clone https://github.com/karthikeyan110505/SMFC_ML_MODELS
2️. Install requirements
pip install -r requirements.txt
3. Run the notebooks or scripts

Train and test models using individual scripts or notebooks.

Results
Voltage trends were successfully predicted using three models.
Model performance was evaluated and compared.
Random Forest achieved highest R² score with lowest MSE.

Future Work
Incorporate more features (e.g., soil parameters, temperature).
Automate real-time monitoring.
Deploy the best model as a web app or IoT dashboard.

Author
Name: Karthikeyan Ravichandran
Contact: karthikeyan110505@gmail.com

If you like this project, give it a ⭐️ and feel free to fork or contribute! 💖✨
