#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:17:14 2021

@author: davidigbokwe
"""

import pandas as pd


df = pd.read_csv("Testcsv.csv") 

df.info()


#column names: iterating the columns 
for col in df.columns: 
    print(col) 


#convert time to integer
df['MonthOriginal'] = pd.to_datetime(df['MonthOriginal'], format = '%m/%d/%Y')
df['MonthOriginal'] = df['MonthOriginal'].astype('int64')//1e9


###decision tree
#split into x and y
x=df.drop(["3MonthsPressureValues"],axis =1)
y=df["3MonthsPressureValues"]

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=98456)


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)

print(dtr.score(x_test,y_test))
""" seed 98456: R2 = 0.****"""

#training
y_pred_train = dtr.predict(x_train)

#test
y_pred_test = dtr.predict(x_test)



from sklearn.metrics import mean_squared_error as mse
mse_dtr = mse(y_test,y_pred_test)
rmse_dtr = mse_dtr**0.5
print("mse is:",mse_dtr)
print("rmse is:",rmse_dtr) 

""" seed 98456
mse is: **.*****
rmse is: *.********"""


##random forest regressor
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 500 decision trees
rfr = RandomForestRegressor(n_estimators = 500, random_state = 98456)


#implement grid search
parameter_grid={"max_depth":range(2,10),"min_samples_split":range(2,8)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(rfr,parameter_grid,verbose=4,scoring="r2", cv =3)

#train 
grid.fit(x_train,y_train)

## best parameters
grid.best_params_

#best max depth 5, min splt 2
parameter_grid={"max_depth": 7,"min_samples_split":2}

# Test
rfr.fit(x_test, y_test)


print(rfr.score(x_test, y_test))
"""R2 score = 0.*******"""
"""seed 96456: R2 score = 0.**********"""


#train
y_pred_train = rfr.predict(x_train)

#test
y_pred_test = rfr.predict(x_test)


from sklearn.metrics import mean_squared_error as mse
mse_rfr = mse(y_test,y_pred_test)
rmse_rfr = mse_rfr**0.5
print("mse is:",mse_rfr)
print("rmse is:",rmse_rfr) 
""" seed 98456
mse is: *.******
rmse is: *.******"""

###

# Put test data in dataframe
df_test = pd.DataFrame(y_test)
df_test.reset_index(level=0, inplace=True)
df_test['id']= range(0,29)


# Put predicted data in dataframe
df_pred = pd.DataFrame(y_pred_test)
df_pred.reset_index(level=0, inplace=True)




combined_withcovid = df_test.merge(df_pred, left_on='id', right_on='index')

combined_withcovid = combined_withcovid.to_csv("Testcsv.csv", sep=',')



#plot 
import matplotlib.pyplot as plt

plt.figure(figsize = (10,10)) 
plt.plot(df_test['id'],df_test['3MonthsPressureValues'], label= "True", color = "red")
plt.plot(df_pred['index'],df_pred[0], label= "Predicted", color = "blue")
plt.title("True vs Predicted", fontsize = 25)
plt.legend(fontsize = 15, loc='upper left')
plt.xlabel("id", fontsize = 20)
plt.ylabel("3MonthsPressureValues", fontsize = 20)
plt.xticks(fontsize=15) 
plt.yticks(fontsize=15) 
plt.show()




### 2nd model




### Removing Covid year

import pandas as pd


df = pd.read_csv("Testcsv.csv")


#convert time to integer
df['MonthOriginal'] = pd.to_datetime(df['MonthOriginal'], format = '%m/%d/%Y')
df['MonthOriginal'] = df['MonthOriginal'].astype('int64')//1e9

#drop 2020
df = df[:-12]

###decision tree
#split into x and y
x=df.drop(["3MonthsPressureValues"],axis =1)

y=df["3MonthsPressureValues"]

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=98456)


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)

print(dtr.score(x_test,y_test))
""" seed 98456: score = *.*******"""

#training
y_pred_train = dtr.predict(x_train)

#test
y_pred_test = dtr.predict(x_test)


from sklearn.metrics import mean_squared_error as mse
mse_dtr = mse(y_test,y_pred_test)
rmse_dtr = mse_dtr**0.5
print("mse is:",mse_dtr)
print("rmse is:",rmse_dtr) 
"""
mse is: **.*****
rmse is: *.********"""

##random forest regressor
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 500 decision trees
rfr = RandomForestRegressor(n_estimators = 500, random_state = 98456)

#implement grid search
parameter_grid={"max_depth":range(2,10),"min_samples_split":range(2,8)}


from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(rfr,parameter_grid,verbose=4,scoring="r2", cv =3)

#train 
grid.fit(x_train,y_train)

## best parameters
grid.best_params_

#best max depth 5, min splt 2
parameter_grid={"max_depth": 5,"min_samples_split":2}

#train
rfr.fit(x_train, y_train)

# Test
rfr.fit(x_test, y_test)


#test
print(rfr.score(x_test, y_test))
"""score = *.*******"""

#training
y_pred_train = rfr.predict(x_train)

#test
y_pred_test = rfr.predict(x_test)

#train data
from sklearn.metrics import mean_squared_error as mse
mse_rfr = mse(y_train,y_pred_train)
rmse_rfr = mse_rfr**0.5
print("mse is:",mse_rfr)
print("rmse is:",rmse_rfr) 
""" seed 98456
mse is: **.*****
rmse is: *.********"""


#test data
from sklearn.metrics import mean_squared_error as mse
mse_rfr = mse(y_test,y_pred_test)
rmse_rfr = mse_rfr**0.5
print("mse is:",mse_rfr)
print("rmse is:",rmse_rfr) 
"""
mse is: *.*****
rmse is: *.********"""


###

# Put test data in dataframe
df_test = pd.DataFrame(y_test)
df_test.reset_index(level=0, inplace=True)
df_test['id']= range(0,26)


# Put predicted data in dataframe
df_pred = pd.DataFrame(y_pred_test)
df_pred.reset_index(level=0, inplace=True)


#plot w/o 2020
import matplotlib.pyplot as plt

plt.figure(figsize = (10,10)) 
plt.plot(df_test['id'],df_test['3MonthsPressureValues'], label= "True", color = "red")
plt.plot(df_pred['index'],df_pred[0], label= "Predicted", color = "blue")
plt.title("True vs Predicted (2020 Removed)", fontsize = 25)
plt.legend(fontsize = 15, loc='upper left')
plt.xlabel("id", fontsize = 20)
plt.ylabel("3MonthsPressureValues", fontsize = 20)
plt.xticks(fontsize=15) 
plt.yticks(fontsize=15) 
plt.show()








#########



combined_without = df_test.merge(df_pred, left_on='id', right_on='index')

combined_without = combined_without.to_csv("Testcsv.csv",sep=',')












