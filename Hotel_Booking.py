##Written by: Sambit & Lingjia

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('hotel_booking_history.csv', header = None)
data.columns = ['Month','Bookings']
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data = data.set_index('Month')


data.plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Hotel guest traffic')



#Mean Imputation to impute missing values

data = data.assign(Bookings_Mean_Imputation=data.Bookings.fillna(data.Bookings.mean()))
data[['Bookings_Mean_Imputation']].plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Hotel guest traffic: Mean imputation')

#Linear Interpolation
data = data.assign(Bookings_Linear_Interpolation=data.Bookings.interpolate(method='linear'))

#Plotting the Graph for Linear Interpolation
plt.legend(loc='best')
plt.title('Hotel guest traffic: Linear interpolation')



data['Bookings'] = data['Bookings_Linear_Interpolation']
data.drop(columns=['Bookings_Mean_Imputation','Bookings_Linear_Interpolation'],inplace=True)


#Outlier Detection

import seaborn as sns
fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=data['Bookings'],whis=1.5)



##Time series decomposition

#Additive Seasonal Decomposition

from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(data.Bookings, model='additive')
fig = decomposition.plot()



#######TIME TO BUILD AND EVALUATE TIME SERIES FORECAST####

train_len = 120
train = data[0:train_len]
test = data[train_len:]

results=0

def NaiveMethod():
    # Naive Method
    y_hat_naive = test.copy()
    y_hat_naive['naive_forecast'] = train['Bookings'][train_len - 1]

    # Plotting for Naive Method
    plt.figure(figsize=(12, 4))
    plt.plot(train['Bookings'], label='Train')
    plt.plot(test['Bookings'], label='Test')
    plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')
    plt.legend(loc='best')
    plt.title('Naive Method')

    ##Calculate RMSE and MAPE for Naive Method
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_naive['naive_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test['Bookings'] - y_hat_naive['naive_forecast']) / test['Bookings']) * 100, 2)
    results = pd.DataFrame({'Method': ['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
    return results


def SAM():
    # Simple Average Method

    y_hat_avg = test.copy()
    y_hat_avg['avg_forecast'] = train['Bookings'].mean()

    # Plotting for Simple Average Forecast
    plt.figure(figsize=(12, 4))
    plt.plot(train['Bookings'], label='Train')
    plt.plot(test['Bookings'], label='Test')
    plt.plot(y_hat_avg['avg_forecast'], label='Simple average forecast')
    plt.legend(loc='best')
    plt.title('Simple Average Method')

    ##Calculate RMSE and MAPE for Simple Average
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_avg['avg_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test['Bookings'] - y_hat_avg['avg_forecast']) / test['Bookings']) * 100, 2)
    tempResults = pd.DataFrame({'Method': ['Simple average method'], 'RMSE': [rmse], 'MAPE': [mape]})
    results = pd.concat([NaiveMethod(), tempResults])
    results = results[['Method', 'RMSE', 'MAPE']]
    return results


def SMAM():
    # Simple Moving Average Method

    y_hat_sma = data.copy()
    ma_window = 12
    y_hat_sma['sma_forecast'] = data['Bookings'].rolling(ma_window).mean()
    y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len - 1]

    # Plotting for Simple Moving Average Forecast

    plt.figure(figsize=(12, 4))
    plt.plot(train['Bookings'], label='Train')
    plt.plot(test['Bookings'], label='Test')
    plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')
    plt.legend(loc='best')
    plt.title('Simple Moving Average Method')

    ##Calculate RMSE and MAPE for Simple Moving Average
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_sma['sma_forecast'][train_len:])).round(2)
    mape = np.round(np.mean(np.abs(test['Bookings'] - y_hat_sma['sma_forecast'][train_len:]) / test['Bookings']) * 100,
                    2)

    tempResults = pd.DataFrame({'Method': ['Simple moving average forecast'], 'RMSE': [rmse], 'MAPE': [mape]})
    results = pd.concat([SAM(), tempResults])
    results = results[['Method', 'RMSE', 'MAPE']]
    return results

def SESM():
    # Simple Exponential Smoothing Method

    import warnings
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    warnings.simplefilter('ignore')

    model = SimpleExpSmoothing(train['Bookings'])
    model_fit = model.fit(smoothing_level=0.2, optimized=False)

    y_hat_ses = test.copy()
    y_hat_ses['ses_forecast'] = model_fit.forecast(24)

    # Plotting for Simple Exponential Smoothing Forecast
    plt.figure(figsize=(12, 4))
    plt.plot(train['Bookings'], label='Train')
    plt.plot(test['Bookings'], label='Test')
    plt.plot(y_hat_ses['ses_forecast'], label='Simple exponential smoothing forecast')
    plt.legend(loc='best')
    plt.title('Simple Exponential Smoothing Method')

    ##Calculate RMSE and MAPE for Simple Exponential Smoothing Forecast
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_ses['ses_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test['Bookings'] - y_hat_ses['ses_forecast']) / test['Bookings']) * 100, 2)

    tempResults = pd.DataFrame({'Method': ['Simple exponential smoothing forecast'], 'RMSE': [rmse], 'MAPE': [mape]})
    results = pd.concat([SMAM(), tempResults])
    results = results[['Method', 'RMSE', 'MAPE']]
    return results

def HMWT():
    # Holt's method with trend

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(np.asarray(train['Bookings']), seasonal_periods=12, trend='additive', seasonal=None)
    model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.01, optimized=False)
    y_hat_holt = test.copy()
    y_hat_holt['holt_forecast'] = model_fit.forecast(len(test))

    # Plotting for Holt's method with trend

    plt.figure(figsize=(12, 4))
    plt.plot(train['Bookings'], label='Train')
    plt.plot(test['Bookings'], label='Test')
    plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s exponential smoothing forecast')
    plt.legend(loc='best')
    plt.title('Holts Exponential Smoothing Method')

    ##Calculate RMSE and MAPE for Holt's method with trend
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_holt['holt_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test['Bookings'] - y_hat_holt['holt_forecast']) / test['Bookings']) * 100, 2)

    tempResults = pd.DataFrame({'Method': ['Holts exponential smoothing method'], 'RMSE': [rmse], 'MAPE': [mape]})
    results = pd.concat([SESM(), tempResults])
    results = results[['Method', 'RMSE', 'MAPE']]
    return results

def HWMMWTAS():
    # Holt Winter's multiplicative method with trend and seasonality

    import warnings
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter('ignore', ConvergenceWarning)

    y_hat_hwm = test.copy()
    model = ExponentialSmoothing(np.asarray(train['Bookings']), seasonal_periods=12, trend='add', seasonal='mul')
    model_fit = model.fit(optimized=True)
    y_hat_hwm['hw_forecast'] = model_fit.forecast(24)

    ##Plot graph for Holt Winter's multiplicative method with trend and seasonality
    plt.figure(figsize=(12, 4))
    plt.plot(train['Bookings'], label='Train')
    plt.plot(test['Bookings'], label='Test')
    plt.plot(y_hat_hwm['hw_forecast'], label='Holt Winters\'s mulitplicative forecast')
    plt.legend(loc='best')
    plt.title('Holt Winters Mulitplicative Method')

    # Calculate RMSE and MAPE
    from sklearn.metrics import mean_squared_error
    y_hat_hwm = y_hat_hwm.assign(hw_forecast1=y_hat_hwm.hw_forecast.fillna(0))
    rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_hwm['hw_forecast1'])).round(2)
    mape = np.round(np.mean(np.abs(test['Bookings'] - y_hat_hwm['hw_forecast1']) / test['Bookings']) * 100, 2)
    tempResults = pd.DataFrame({'Method': ['Holt Winters multiplicative method'], 'RMSE': [rmse], 'MAPE': [mape]})
    results = pd.concat([HMWT(), tempResults])
    results = results[['Method', 'RMSE', 'MAPE']]
    return results

results=HWMMWTAS()


# AutoRegressive Methods

data['Bookings'].plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Hotel Guest Traffic')

from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(data['Bookings'])
import warnings
from statsmodels.tsa.stattools import kpss
warnings.simplefilter('ignore')
from statsmodels.tsa.stattools import kpss
kpss_test = kpss(data['Bookings'])
from scipy.stats import boxcox
data_boxcox = pd.Series(boxcox(data['Bookings'], lmbda=0), index=data.index)

plt.figure(figsize=(12, 4))
plt.plot(data_boxcox, label='After Box Cox tranformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), data.index)
plt.figure(figsize=(12, 4))
plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')
plt.legend(loc='best')
plt.title('After Box Cox transform and differencing')
data_boxcox_diff.dropna(inplace=True)
adf_test = adfuller(data_boxcox_diff)
import warnings
from statsmodels.tsa.stattools import kpss

kpss_test = kpss(data_boxcox_diff)
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(12, 4))
plot_acf(data_boxcox_diff, ax=plt.gca(), lags=30)
from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize=(12, 4))
plot_pacf(data_boxcox_diff, ax=plt.gca(), lags=30)
train_data_boxcox = data_boxcox[:train_len]
test_data_boxcox = data_boxcox[train_len:]
train_data_boxcox_diff = data_boxcox_diff[:train_len - 1]
test_data_boxcox_diff = data_boxcox_diff[train_len - 1:]



from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train_data_boxcox_diff, order=(1, 0, 0))
model_fit = model.fit()
y_hat_arima = data_boxcox_diff.copy()
y_hat_arima['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arima['ar_forecast_boxcox'] = y_hat_arima['ar_forecast_boxcox_diff'].cumsum()
y_hat_arima['ar_forecast_boxcox'] = y_hat_arima['ar_forecast_boxcox'].add(data_boxcox[0])
y_hat_arima['ar_forecast'] = np.exp(y_hat_arima['ar_forecast_boxcox'])
plt.figure(figsize=(12, 4))
plt.plot(train['Bookings'], label='Train')
plt.plot(test['Bookings'], label='Test')
plt.plot(y_hat_arima['ar_forecast'][test.index.min():], label='Auto regression forecast')
plt.legend(loc='best')
plt.title('Auto Regression Method')
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_arima['ar_forecast'][test.index.min():])).round(2)
mape = np.round(
np.mean(np.abs(test['Bookings'] - y_hat_arima['ar_forecast'][test.index.min():]) / test['Bookings']) * 100, 2)

tempResults = pd.DataFrame({'Method': ['Autoregressive (AR) method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]


#SARIMAX Model
import warnings
warnings.simplefilter('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train_data_boxcox, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=0)
y_hat_sarima = data_boxcox_diff.copy()
y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])

##Plotting the graph for SARIMAX Model
plt.figure(figsize=(12,4))
plt.plot(train['Bookings'], label='Train')
plt.plot(test['Bookings'], label='Test')
plt.plot(y_hat_sarima['sarima_forecast'][test.index.min():], label='SARIMA forecast')
plt.legend(loc='best')
plt.title('Seasonal autoregressive integrated moving average (SARIMA) method')

#Calculating RMSE and MAPE for SARIMAX model

rmse = np.sqrt(mean_squared_error(test['Bookings'], y_hat_sarima['sarima_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['Bookings']-y_hat_sarima['sarima_forecast'][test.index.min():])/test['Bookings'])*100,2)

tempResults = pd.DataFrame({'Method':['Seasonal autoregressive integrated moving average (SARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
print(results)
plt.show()
