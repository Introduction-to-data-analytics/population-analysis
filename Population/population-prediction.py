import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

df = pd.read_csv('Population_resize.csv')
df2 = pd.read_csv('new.csv')

usage = df.drop(['Indicator Name','Indicator Code','1960','2014','2015','2016'],1)

uni_df = list(set(list(df.loc[:,'Country Name'])))
uni_df2 = list(set(list(df2.loc[:,'Region'])))
country = []
for country1 in uni_df:
    for country2 in uni_df2:
        if country1 == country2:
            country.append(country2)
            print(country2)
print(len(country))
population = usage.drop(['Country Name', 'Country Code', '2013'],1)

target = df['2013']

#for i in range(len(population)):
#    plt.figure()
#    plt.ylabel('value')
#    plt.xlabel(df.iloc[i,0])
#    plt.boxplot(population.iloc[i,:])
#    
def corr(ser):
    corr_value = target.to_frame().join(ser.to_frame()).corr()
    print(corr_value.index[1])
    print(corr_value.iloc[0,1])
    
#population.apply(corr)
#
#X = np.array(population)
#y = np.array(target)
#
###10-fold cross validation
#kf = KFold(n_splits=10)
#lm = linear_model.LinearRegression()
#count = 1
#for train_index, test_index in kf.split(X):
#    #Divide data in to 2 sections(train and test section)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    
#    #Fit data in to linear regression model
#    lm.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))
#    
#    print('cross validation No.%d' %(count))
#    
##    #Model
##    model = make_model(lm.intercept_,lm.coef_[0][0],lm.coef_[0][1],lm.coef_[0][2])
##    print(model)
#    
#    #Predict y value by above linear regression
#    y_predicted= lm.predict(pd.DataFrame(X_test))
#    y_predicted_df = pd.DataFrame(y_predicted).rename(columns={0:'y_predicted'})
#    y_test_df = pd.DataFrame(y_test).rename(columns={0:'y_test'})
#    print(y_predicted_df.join(y_test_df))
#    
#    #Calculate Root Mean Squear Error(RMSE)
#    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
#    print('RMSE : %f\n' %(rmse))
#    count += 1
#
#lm.fit(X, y)
#
#output = []
#output.append(lm.intercept_)
#output = output+list(lm.coef_)
#output = pd.DataFrame(output).rename(columns={0:'inter and cofe'})
#output.to_csv('population_model.csv', sep=',', index=False)
#
#predict = '2025'
#
#y_predicted= lm.predict(X)#predict 2013
#y_predicted_df = pd.DataFrame(y_predicted).rename(columns={0:'2013'})
#parameter = population.iloc[:,1:52].join(y_predicted_df)
#
#for i in range(int(predict)-2013): 
#    X = np.array(parameter)
#    y_predicted= lm.predict(X)
#    y_predicted_df = pd.DataFrame(y_predicted).rename(columns={0:str(2013+i+1)})
#    if 2013+i+1 > 2016:
#        df = df.join(y_predicted_df)
#    parameter = parameter.iloc[:,1:52].join(y_predicted_df)
#    print(parameter)
#
#df.to_csv('population_predicted.csv', sep=',', index=False)
#
#result = df[['Country Name']].join(parameter.loc[:,[predict]])
#selected = result.loc[result['Country Name'] == 'Thailand']
#print(result)
#print(selected)