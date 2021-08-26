# %% [markdown]
# author: Saul Chirinos
# %%
# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse as RMSE

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

import pmdarima.arima as pm

# Set plotting style
sns.set()

# Format output numbers
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# %%
# Load datasets
# Historical data
panama = pd.read_csv('Data/continuous dataset.csv',
                     parse_dates=['datetime'],
                     index_col='datetime')

# Official forecast data from CND
dispatch = pd.read_csv('Data/weekly pre-dispatch forecast.csv',
                       parse_dates=['datetime'],
                       index_col='datetime')
panama.index
# %%
# Set the datetime frequency to hourly
panama = panama.asfreq('H')
# %%
###########################################################
################ EXPLORATORY DATA ANALYSIS ################
# View first 5 rows
panama.head()
# %%
# View last 5 rows
panama.tail()
# %%
# Rows and columns
panama.shape
# %%
# Descriptive statistics
panama.describe().transpose().round(2)
# %%
# Check for missing data
panama.isnull().sum()
# %%
# Time horizon demand plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

panama.nat_demand.plot(ax=ax1)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Hourly Load (MWh)', fontsize=12)
ax1.set_title('Panama Electricity Load', size=16)
ax1

panama[:'2015'].nat_demand.plot(ax=ax2)
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Hourly Load (MWh)', fontsize=12)
ax2.set_title('Panama Electricity Load in 2015', size=16)

panama[:'2015-01'].nat_demand.plot(ax=ax3)
ax3.set_xlabel('Day', fontsize=12)
ax3.set_ylabel('Hourly Load (MWh)', fontsize=12)
ax3.set_title('Panama Electricity Load on January, 2015', size=16)

panama[:'2015-01-03'].nat_demand.plot(ax=ax4)
ax4.set_xlabel('Hour', fontsize=12)
ax4.set_ylabel('Hourly Load (MWh)', fontsize=12)
ax4.set_title('Panama Electricity Load on January 3, 2015', size=16)

plt.show()
# %%
# Zoom in on sharp demand drops
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18))

panama['2017-06-23':'2017-07-05'].nat_demand.plot(ax=ax1)
ax1.set_xlabel('Day', fontsize=12)
ax1.set_ylabel('Hourly Load (MWh)', fontsize=12)
ax1.set_title(
    'Panama Electricity Load between June 23, 2017 & July 5, 2017', size=16)

panama['2019-01-15':'2019-01-25'].nat_demand.plot(ax=ax2)
ax2.set_xlabel('Day', fontsize=12)
ax2.set_ylabel('Hourly Load (MWh)', fontsize=12)
ax2.set_title(
    'Panama Electricity Load between January 15, 2019 & January 25, 2019', size=16)
plt.show()
# %%
# ADF test check for stationarity
# Null: Series has a unit root (not stationary)

# Level series
result = adfuller(panama.nat_demand, autolag='AIC')
print('ADF Statistic:', result[0].round(2),
      '\np-value:', result[1])

# Differenced series
result = adfuller(panama.nat_demand.diff().dropna(), autolag='AIC')
print('ADF Statistic:', result[0].round(2),
      '\np-value:', result[1])
# %%
# KPSS test check for stationarity
# Null: Series is stationary

# Level series
result = kpss(panama.nat_demand)
print('KPSS Statistic:', result[0].round(2),
      '\np-value:', result[1])

# First differenced series
result = kpss(panama.nat_demand.diff().dropna())
print('KPSS Statistic:', result[0].round(2),
      '\np-value:', result[1])
# %%
# Estimate the number of non-seasonal and seasonal differences
print('Estimated number of non-seasonal differences:',
      pm.ndiffs(panama.nat_demand))

print('Estimated number of seasonal differences:',
      pm.nsdiffs(panama.nat_demand, m=24))
# %%
# Plot the ACF and PACF
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Level series
plot_acf(panama.nat_demand, lags=60, ax=ax1)
plot_pacf(panama.nat_demand, lags=60, ax=ax2)

# Differenced series
plot_acf(panama.nat_demand.diff().dropna(), lags=60, ax=ax3)
plot_pacf(panama.nat_demand.diff().dropna(), lags=60, ax=ax4)
plt.show()
# %%
# Seasonally decompose series into parts and plot
# Level series
decomposition1 = seasonal_decompose(panama[:'2015-01-08'].nat_demand,
                                    period=24)
decomposition1.plot()
plt.show()

# Differenced series
decomposition2 = seasonal_decompose(panama[:'2015-01-08'].nat_demand.diff().dropna(),
                                    period=24)
decomposition2.plot()
plt.show()
################ EXPLORATORY DATA ANALYSIS ################
###########################################################
# %%
############################################################
#################### DATA PREPROCESSING ####################
# Filter for the last month of data (June, 2020) for modeling
panama = panama['2020-06':]
# Filter for plotting
dispatch = dispatch['2020-06-20 01:00:00':'2020-06-27 00:00:00']
# Transform dependant variable
panama['nat_demand'] = np.log(panama.nat_demand)
#################### DATA PREPROCESSING ####################
############################################################
# %%
###########################################################
#################### FEATURE SELECTION ####################
# Check distributions of variables
float_cols = [col for col in panama.columns if panama[col].dtype == float]
# Removes national demand feature
float_cols.remove('nat_demand')

fig, ax = plt.subplots(4, 3, figsize=(10, 16))
# Loop through each chart in the subplot
for count, item in enumerate(ax.reshape(-1)):
    col = float_cols[count]
    # Histogram for each continuous X variable
    item.hist(panama[col], bins=20)
    item.set_xlabel(col)
plt.show()
# %%
# Plot the correlation matrix
plt.figure(figsize=(15, 10))
corr = panama.corr(method='spearman')
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='vlag_r')
plt.show()
# %%
# Find most important features by Random Forest
X = panama.drop(columns='nat_demand')
y = panama.nat_demand

rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
importances = pd.Series(data=rf.feature_importances_,
                        index=X.columns).sort_values()

importances.plot(kind='barh', figsize=(10, 8))
plt.title('Features Importance')
plt.show()
#################### FEATURE SELECTION ####################
###########################################################
# %%
###########################################################
####################### FINAL MODEL #######################
# Split into training and testing sets
train_df = panama.iloc[:len(panama) - 168]
test_df = panama.iloc[len(panama)-168:]

X = ['T2M_dav']
y = train_df.nat_demand

model = SARIMAX(y, order=(1, 0, 2), seasonal_order=(2, 0, 2, 24),
                trend='c', exog=train_df[X])
results = model.fit()

results.plot_diagnostics(figsize=(16, 16))
plt.show()

print(results.summary())
# %%
# Forecast the last week (168 hours) of data
predictions = results.forecast(steps=168, exog=test_df[X])

# Return values back to scale for plotting
test_df['nat_demand'] = np.exp(test_df.nat_demand)
predictions = np.exp(predictions)

# Evaluation metrics
mape = MAPE(test_df.nat_demand, predictions)
mae = MAE(test_df.nat_demand, predictions)
mse = MSE(test_df.nat_demand, predictions)
rmse = RMSE(test_df.nat_demand, predictions)

print('MAPE =', mape.round(3))
print('MAE =', mae.round(2))
print('MSE =', mse.round(2))
print('RMSE =', rmse.round(2))
# %%
# Plots
# Actual vs. Forecast
plt.figure(figsize=(16, 12))
plt.plot(test_df.index, test_df.nat_demand, label='Actual')
plt.plot(predictions.index, predictions, label='Forecast')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Hourly Load (MWh', fontsize=16)
plt.title('Actual Load vs. SARMAX(1, 2)x(2, 2, 24)', fontsize=20)
plt.legend()
plt.show()

# Actual vs. Forecast vs. CND
plt.figure(figsize=(16, 12))
plt.plot(test_df.index, test_df.nat_demand, label='Actual')
plt.plot(predictions.index, predictions, label='Forecast')
plt.plot(dispatch.index, dispatch[['load_forecast']], label='CND')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Hourly Load (MWh', fontsize=16)
plt.title('Actual Load vs. SARMAX(1, 2)x(2, 2, 24) vs. CND Forecast', fontsize=20)
plt.legend()
plt.show()
#################### FINAL MODEL #######################
########################################################
# %%
#######################################################
#################### MODEL TESTING ####################
# Load datasets
panama = pd.read_csv('Data/continuous dataset.csv',
                     parse_dates=['datetime'],
                     index_col='datetime')
dispatch = pd.read_csv('Data/weekly pre-dispatch forecast.csv',
                       parse_dates=['datetime'],
                       index_col='datetime')

# Set frequency of samples
panama = panama.asfreq('H')
# Filter for the last month of data
panama = panama['2020-06':]
# Filter for plotting
dispatch = dispatch['2020-06-20 01:00:00':'2020-06-27 00:00:00']
# Transform dependant variable
panama['nat_demand'] = np.log(panama.nat_demand)

# Split into training and testing sets
train_df = panama.iloc[:len(panama) - 168]
test_df = panama.iloc[len(panama)-168:]
# %%
# Features to test
#X = ['T2M_dav', 'QV2M_toc', 'W2M_dav']
#X = ['T2M_dav', 'QV2M_toc']
#X = ['T2M_dav', 'W2M_dav']
X = ['T2M_dav']

# Model testing
# Play around with the numbers and arguments
model = pm.auto_arima(train_df.nat_demand, train_df[X],
                      start_p=0, max_p=3, d=0, start_q=0,
                      max_q=3, seasonal=True, m=24,
                      trace=True)
model.summary()
#################### MODEL TESTING ####################
#######################################################
