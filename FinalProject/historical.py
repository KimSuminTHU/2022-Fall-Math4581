
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from tabulate import tabulate

data = fdr.DataReader('AAPL', '2010-01-01', '2022-12-01')
data = data[['Adj Close']]

#calculate daily return and drop the missing data
returns_daily = data.pct_change()
returns_daily = returns_daily.dropna()

#get VaR
# VaR구하기
returns_daily.sort_values('Adj Close',inplace=True)
q_90 = returns_daily['Adj Close'].quantile(0.10)
q_95 = returns_daily['Adj Close'].quantile(0.05)
q_99 = returns_daily['Adj Close'].quantile(0.01)

print(tabulate([['90%',q_90],['95%',q_95],['99%',q_99]],headers = ['Confidence level','VaR']))


plt.style.use('seaborn') #스타일

#막대그래프
plt.hist(returns_daily['Adj Close'],bins = 40)
#신뢰수준 표시
plt.axvline(x=q_90, color='orange',  ls='--', ymin=0, ymax=0.8, label='90%')
plt.axvline(x=q_95, color='purple', ls='--', ymin=0, ymax=0.8, label='95%')
plt.axvline(x=q_99, color='black',  ls='--', ymin=0, ymax=0.8, label='99%')

plt.title("VaR - Historical Method")
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
plt.savefig('var_historical.jpg')
plt.show()

mean = np.mean(returns_daily['Adj Close'])
std = np.std(returns_daily['Adj Close'])
