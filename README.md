# Predictive-Maintenance-of-Aircraft-Engine-using-CNN-LSTM-and-Attention-Model
This project aims to develop machine learning models for predicting the remaining useful life (RUL) of aircraft engines based on sensor data collected during operation cycles. Accurate RUL forecasting allows for optimization of maintenance scheduling through a predictive/condition-based approach.

# Background
Traditional aircraft engine maintenance is done at fixed time intervals irrespective of the actual condition or degradation of engines. This leads to unnecessary downtime and costs when engines could have been used for longer before maintenance was needed. With advancement in sensor technologies, critical engine parameters like pressure, temperature, vibration etc. can now be continuously monitored. This generates large datasets containing signatures of equipment degradation over cycles of use. If these cycle-sequence datasets are analyzed using machine learning, the current health state and RUL can be estimated. This enables transition from schedule-based to condition-based/predictive maintenance for optimizing service.

# Problem Statement
Given sensor reading data collected from aircraft engines during operational cycles, develop deep learning models that can accurately predict the Remaining Useful Life (RUL) of an engine based on its current cycle readings.

# Data
The dataset contains sensor measurements from 100 aircraft engines recorded every cycle until end of useful life. There are 21 sensor variables and each engine has a different number of maximum cycles (ranging from hundreds to thousands). Additional identifiers include engine ID, cycle number and operational settings. The target variable is the RUL which is calculated as the difference between current cycle and maximum cycles for that engine ID.

# Methodology
The main steps followed are:
**Data Loading and Preprocessing:** Cleaning missing/error values, feature selection, normalization, adding RUL target column
**Exploratory Data Analysis:** Visualizing patterns, correlation analysis to understand degradation signals
**Feature Engineering:** Deriving temporal/lag features to capture trends over cycles
**Model Development:** RNN, CNN and other deep learning architectures for sequence forecasting
**Hyperparameter Tuning:** Grid search CV for optimal configurations
**Model Evaluation:** Performance metrics on test set of unseen engine instances
**Results and Discussion:** Analysis of prediction accuracies, strengths/limitations

# Implementation
The project is implemented in Python leveraging libraries like Pandas, NumPy, PyTorch, Keras and Scikit-learn. Jupyter Notebooks provide an interactive environment for EDA, model building and analysis.

# Conclusion
Preliminary results indicate deep learning approaches can accurately forecast RUL given current cycle data, outperforming statistical benchmarks. With further refinements, these models can enable effective predictive maintenance planning and reduce maintenance costs for aircraft operators.
