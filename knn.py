#!/usr/bin/python
#-------------------------------------------------------------------------------
# Blast Analytics
# KNN test
#-------------------------------------------------------------------------------

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#-------------------------------------------------------------------------------
 
def main():

#READ DATA
    dfr = pd.read_csv('data/regions.csv')
    df = dfr.dropna()

#A BIT OF EXPLORATION
    #print(df.head())
    #print(df.isnull().values.any())
    #print(df['Region'].unique())

#EDA - all pairs
    #plt.close();
    #sns.set_style("whitegrid");
    #sns.pairplot(df, hue="Region", height=3,diag_kws={'bw': 1.5});
    #plt.show()

#EDA - individual scatter
    #plt.close();
    #sns.set_style("whitegrid");
    #sns.FacetGrid(df, hue='Region', height=5) \
    #.map(plt.scatter, "Region", "Population Total") \
    #.add_legend();
    #plt.show()

#save previous plots
    #.map(plt.scatter, "Region", "Birth Rate") \
    #.map(plt.scatter, "Region", "CO2 Emissions") \
    #.map(plt.scatter, "Region", "Energy Usage") \
    #.map(plt.scatter, "Region", "GDP") \
    #.map(plt.scatter, "Region", "Health Exp/Capita") \
    #.map(plt.scatter, "Region", "Lending Interest") \
    #.map(plt.scatter, "Region", "Life Expectancy Female") \
    #.map(plt.scatter, "Region", "Life Expectancy Male") \
    #.map(plt.scatter, "Region", "Population Total") \

#NORMALIZATION
    x_data = df.drop(['Region'],axis=1)
    y_data = df['Region']
    MinMaxScaler = preprocessing.MinMaxScaler()
    X_data_minmax = MinMaxScaler.fit_transform(x_data)
    data = pd.DataFrame(X_data_minmax,columns=['Birth Rate','CO2 Emissions','Energy Usage','GDP','Health Exp/Capita','Lending Interest','Life Expectancy Female','Life Expectancy Male','Population Total'])

#TRIM TO TEST INDIVIDUAL FEATURES
    #data = data[['Birth Rate']]
    #data = data[['O2 Emissions']]

#TRAIN
    X_train, X_test, y_train, y_test = train_test_split(data, y_data,test_size=0.2, random_state = 1)
    knn_clf=KNeighborsClassifier()
    knn_clf.fit(X_train,y_train)
    ypred=knn_clf.predict(X_test) 

#MODEL PARAMETERS
    print('Params')
    print(knn_clf.get_params())

#RESULTS
    print('Matrix 1 - Training/Testing')
    result = confusion_matrix(y_test, ypred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, ypred)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(y_test,ypred)
    print("Accuracy:",result2)

#READ UNKNOWN DATA
    dfu = pd.read_csv('data/unknown.csv')
    #dfu = dfr.dropna()

#PREPEND UNKNOWN DATA -- NOW ORIGINAL DATA PLUS ONE ROW
    dfp = dfu.append(df, ignore_index=True)
    
#NORMALIZATION
    x_data = dfp.drop(['Region'],axis=1)
    y_data = dfp['Region']
    MinMaxScaler = preprocessing.MinMaxScaler()
    X_data_minmax = MinMaxScaler.fit_transform(x_data)
    data2 = pd.DataFrame(X_data_minmax,columns=['Birth Rate','CO2 Emissions','Energy Usage','GDP','Health Exp/Capita','Lending Interest','Life Expectancy Female','Life Expectancy Male','Population Total'])


#CREATE NEW SINGLE ENTRY
    data3 = pd.DataFrame(columns=['Birth Rate','CO2 Emissions','Energy Usage','GDP','Health Exp/Capita','Lending Interest','Life Expectancy Female','Life Expectancy Male','Population Total'])
    data3.loc[0] = data2.iloc[0]
    print("Only one row:")
    print(data3)

#PREDICTIONS
    zpred=knn_clf.predict(data3) 
    print("The predicted region is:")
    print(zpred)

#VISUALIZE IT
    print(dfp.head())
    sns.set_style("whitegrid");
    sns.FacetGrid(dfp, hue='Region', height=5) \
    .map(plt.scatter, "Region", "Energy Usage") \
    .add_legend();
    plt.show()

    #.map(plt.scatter, "Region", "GDP") \
    #.map(plt.scatter, "Region", "Birth Rate") \
    #.map(plt.scatter, "Region", "CO2 Emissions") \
    #.map(plt.scatter, "Region", "Health Exp/Capita") \
    #.map(plt.scatter, "Region", "Lending Interest") \
    #.map(plt.scatter, "Region", "Life Expectancy Female") \
    #.map(plt.scatter, "Region", "Life Expectancy Male") \
    #.map(plt.scatter, "Region", "Population Total") \

#-------------------------------------------------------------------------------
 
if __name__ == '__main__':
    main()
 
#-------------------------------------------------------------------------------


 
