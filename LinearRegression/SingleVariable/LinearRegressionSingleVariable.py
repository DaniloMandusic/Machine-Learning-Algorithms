'''
task: predict what will be the income per capita in canada in year 2020 based on prevous income data

libraries: sklearn - linear_model

steps:
    - read csv - Done
    - plot csv - Done
    - separate columns - Done
    - create linear regression object - Done
    - predict 2020 value - Done
    - write down slope and intercept - Done
    - create new csv with prediction data - Done
    - plot solution -
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# read csv

columnList = ["year", "income"]
csvData = pd.read_csv('Sources/income_per_capita_canada.csv', usecols=columnList)
#print(csvData)

# plot csv

plt.xlabel("year")
plt.ylabel("income")
plt.scatter(csvData["year"], csvData["income"], color='red', marker='+')
#plt.show()

# separate columns

newCsvData = csvData.drop('income', axis='columns') # years 2d array
incomes = csvData.income

#print(newCsvData)
#print(incomes)

# create linear regression object

reg = linear_model.LinearRegression()
reg.fit(newCsvData, incomes)

result = reg.predict([[2020]])
#print(result[0])

# slope and intercept

slope = reg.coef_
intercept = reg.intercept_

#print("function: y = " + str(slope[0]) + "*x + " + str(intercept))

# new csv with prediction data

valuesForPrediction = pd.read_csv("Sources/values_to_predict.csv")
prediction = reg.predict(valuesForPrediction)

#print(prediction)

valuesForPrediction["income"] = prediction
#print(valuesForPrediction)

#valuesForPrediction.to_csv("Sources/prediction.csv")

# plot result

plt.scatter(csvData["year"], csvData["income"], color='red', marker='+')

plt.plot(csvData["year"], slope*csvData["year"] + intercept)
#plt.show()
