import numpy as np
from tabulate import tabulate
import FinanceDataReader as fdr


data = fdr.DataReader('AAPL', '2010-01-01', '2022-12-01')
data = data[['Adj Close']]

TRADING_DAYS = data.shape[0]

# Log revenue
data['Log Rets'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))

# daily standard deviation and annual volatility
daily_vol = np.std(data['Log Rets'])
vol = daily_vol * TRADING_DAYS ** 0.5

t = 1 #risk after t day

def MC_VaR(er, vol, T, iterations):
    end = np.exp((er * vol ** 2) * T +
                     vol * np.sqrt(T) * np.random.standard_normal(iterations))
    return end

at_risk = MC_VaR(er=0.25, vol=vol, T=t/TRADING_DAYS, iterations=50000) -1

percentiles = [1,5,10]
v99,v95,v90 = np.percentile(at_risk, percentiles)

print(tabulate([['90%',v90],['95%',v95],['99%',v99]],headers = ['Confidence level','VaR']))

#graph
import matplotlib.pyplot as plt
plt.style.use('seaborn') #스타일

#bar graph
plt.hist(at_risk,bins = 40)

#confidence level
plt.axvline(x=v99, color='black',  ls='--', ymin=0, ymax=0.8, label='90%')
plt.axvline(x=v95, color='purple', ls='--', ymin=0, ymax=0.8, label='95%')
plt.axvline(x=v90, color='orange',  ls='--', ymin=0, ymax=0.8, label='99%')

plt.title("VaR - Monte Carlo Simulation")
plt.xlabel('Returns')
plt.ylabel('Count of Simulations')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.show()

