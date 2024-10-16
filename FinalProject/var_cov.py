# 라이브러리 불러오기
import numpy as np
import FinanceDataReader as fdr
from tabulate import tabulate
from scipy.stats import norm

# get data from FinanceDataReader library
data = fdr.DataReader('AAPL', '2010-01-01', '2022-12-01')
data = data[['Adj Close']]


# calculate daily revenue / drop na values
returns_daily = data.pct_change()
returns_daily = returns_daily.dropna()

#mean, std
mean = np.mean(returns_daily['Adj Close'])
std = np.std(returns_daily['Adj Close'])

# VaR
returns_daily.sort_values('Adj Close',inplace=True)
var_90 = norm.ppf(1-0.90,mean,std) #ppf : Percent point function
var_95 = norm.ppf(1-0.95,mean,std)
var_99 = norm.ppf(1-0.99,mean,std)

#result
print(tabulate([['90%',var_90],['95%',var_95],['99%',var_99]],headers = ['Confidence level','VaR']))

#graph
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('seaborn')

plt.hist(returns_daily['Adj Close'],bins = 40, density = True)
x = np.linspace(mean-3*std,mean+3*std,100)
plt.plot(x,norm.pdf(x,mean,std),"r") #pdf : Probability density function

#mark confidence level
plt.axvline(x=var_90, color='orange', ls='--', ymin=0, ymax=0.8, label='90%')
plt.axvline(x=var_95, color='purple', ls='--', ymin=0, ymax=0.8, label='95%')
plt.axvline(x=var_99, color='black', ls='--', ymin=0, ymax=0.8, label='99%')

plt.title("VaR - Variance Covariance Method")
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.show()