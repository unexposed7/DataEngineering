#!/usr/bin/env python
# coding: utf-8

import numpy as np              # importing Data Analysis Libraries
import pandas as pd
import random
from IPython.display import Image

n = np.random.randint(0, 1000)  # Generating random values
m = np.random.randint(0, 1000)
k = np.random.randint(0, 1000)
N = np.random.randint(0, 1000)

print("Values randomly generated, n=",str(n),"m=",str(m),"k=",str(k),"N=",str(N))


# ## Initialization of the values and creating My-Data table
# ### Selecting the size of the attributes and populating them.. for example Feature_A <span style="color:blue"> [FA-1, ..FA-n] </span> is until  n=950   
# ### for Attributes Features FB & FC ->   <span style="color:blue"> FB-1,FB-2,..FB-m </span>until m and <span style="color:green"> FC-1,FC-2,..FC-k</span> until k


# Creating a list of Features_A for the Feature Name until n (until m for Feature B and k for Feature C)
# then the Feature A value which is: FA-1,FA-2,...FA-n
# then the Total for each Feature which is sum(N) for Feature A and C and count(ID) for Feature B
# then the percentage of all values devided by Total
# and then the normalization of percentage - explained below!
# percentage rounded to the 3rd decimal
# adding all of them in one Dataframe (Feature_FA)
# and renaming the column names 

Feature_FA        = pd.DataFrame(["Feature_A" for id in range(1,n)])           
FA_string         = pd.DataFrame(["FA-"+str(id) for id in range(1,n)])         
TotalFA           = pd.DataFrame([random.randrange(1,1000) for i in range(n)]) 
percentageFA      = TotalFA/TotalFA.sum()                                   
percentageFAR     = percentageFA.round(decimals=3)
percentageFAnorm  = 1/((percentageFA.max()-percentageFA.min())*(percentageFA-percentageFA.max())+1)/len(percentageFA)
percentageFAnormR = percentageFAnorm.round(decimals=3)                                                                     


Feature_FA["FA_string"]  = FA_string
Feature_FA["TotalFA"]    = TotalFA
Feature_FA["Percentage"] = percentageFAR
Feature_FA.columns       ={"Feature_Name":"Feature_Name",
                           "Feature_Value":"Feature_Value",
                            "Total":"Total",
                            "Percentage":"Percentage"}

# same process follows for Feature B

Feature_FB        = pd.DataFrame(["Feature_B" for id in range(1,m)])
FB_string         = pd.DataFrame(["FB-"+str(id) for id in range(1,m)])
TotalFB           = pd.DataFrame([random.randrange(1,1000) for i in range(m)])
percentageFB      = TotalFB/TotalFB.sum()
percentageFB      = percentageFB.round(decimals=3)


Feature_FB["FB_string"]  = FB_string
Feature_FB["TotalFB"]    = TotalFB
Feature_FB["Percentage"] = percentageFB
Feature_FB.columns       = {"Feature_Name":"Feature_Name",
                            "Feature_Value":"Feature_Value",
                            "Total":"Total",
                            "Percentage":"Percentage"}

## same process followed for Feature C

Feature_FC       = pd.DataFrame(["Feature_C" for id in range(1,k)])
FC_string        = pd.DataFrame(["FC-"+str(id) for id in range(1,k)])
TotalFC          = pd.DataFrame([random.randrange(1,1000) for i in range(k)])
percentageFC     = TotalFC/TotalFC.sum()
percentageFC     = percentageFC.round(decimals=3)

Feature_FC["FC_string"]  = FC_string
Feature_FC["TotalFC"]    = TotalFC
Feature_FC["Percentage"] = percentageFC
Feature_FC.columns       = {"Feature_Name":"Feature_Name",
                            "Feature_Value":"Feature_Value",
                            "Total":"Total",
                            "Percentage":"Percentage"}

# Adding all 3 features to ResultTable using the append mathod

ResultTable = pd.DataFrame(Feature_FA)            
ResultTable = ResultTable.append(Feature_FB)
ResultTable = ResultTable.append(Feature_FC)
ResultTable = ResultTable.set_index([pd.Index(range(n+m+k-3))])


# The sum(N) (for features A & C) and count(Id) (for feature B) to calculate the Total(FA,FB,FC) 
# was not calculated for each feature value but instead was added from randonly genarated numbers
# this is the value Total(FA,FB,FC) --> TotalFC=pd.DataFrame([random.randrange(1,1000) for i in range(k)])
# I guess there is a small ambiguity in the exercise, because:
# If each Feature had an attribute N, then the sum(N), would have created 2 different values for the 2 different features, 
# since the sum of N would be the same and since Total=sum(N), (For A & C)


Id_desc = "An Id which is also the key to the table MyData"
Feature_desc = "A feature."
N_desc = "An extensive quantity such as count, lenth, volume..."


data = {"Attribute": ["Id", "Feature_FA", "Feature_FB", "Feature_FC", "N"], "Type": ["Type(Id)",str,str,str,int],
        "Domain": ["Type(Id)",FA_string, FB_string, FC_string, "N>=0"],
        "Description":[Id_desc,Feature_desc,Feature_desc,Feature_desc,N_desc]}
MyDataTable = pd.DataFrame(data=data)


print(MyDataTable) # Preview of My-Data table,  same as the exercise


# ### saving data to csv file 
# #### the file can be found at git repo

MyDataTable.to_csv(r'C:\Users\Admin\Dropbox\Machine_Learning\ML2020\DataEngineeringTest\MyData.csv')


# ### Printing the Result Table


print(ResultTable.head())


print(ResultTable.tail())


#### To make sure that the sum of all percetages (for all features) is always equal to 1
#### by deviding with the summary of all percentages, this is achieved

#### Furthermore, the way to perform this is by feature scaling - data normalization
#### with min-max scaling and rescaling the range of features to the range in [0, 1]

#### Onother way to linearly rescale data values into a new range min' to max' is
#### by using the following formula:
#### normilized= (max'-min')/((max-min)*(value-max)+max'/ #number of values), 
#### where in our case, max' =1

#### In order to present the different cases, and the rounding error 
#### all values were preserved (only for Feature A, for demostration purposes)

print("Sum of all percentages devided by sum:",percentageFA.sum(),"\n",
      "Sum of all percentages rounded id 3rd decimal",percentageFAR.sum(),"\n",
      "Sum of all percentages normilized:",percentageFAnorm.sum(),"\n",
      "Sum of all percentages rounded in 3rd decimal after normzalization",percentageFAnormR.sum())


####  By rounding in the 3rd decimal, we lose in accuracy
####  This is a numeric decision problem on normalization and can be further discussed
#### Some interesting information could be found under the following link
#   https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html


#### ANother was to perform feature scaling is by using reade libray modules
#### importing the preprocessing module of the machine learning library skikit-learn
#### as follows


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(percentageFA)
scaler


print(percentageFA.sum(), percentageFB.sum(), percentageFC.sum())


# ### Creating MyData table in SQL


import sqlite3

connection = sqlite3.connect("MyDataTable.db")
# connection where saved db is stored - MyDataTable.db
cursor = connection.cursor()

sql_command ="""DROP TABLE IF EXISTS MyDataTable;"""
cursor.execute(sql_command)

sql_command ="""CREATE TABLE MyDataTable(
    Id CHAR(8) PRIMARY KEY, 
    Feature_Name TEXT, 
    Feature_Value CHAR(8),
    Total INTEGER, 
    N INTEGER,
    Percentage FLOAT);"""

cursor.execute(sql_command) 


sql_command = """INSERT INTO MyDataTable(
    Id, Feature_Name, Feature_Value, Total, N, Percentage) 
    VALUES(abs(random()), "Feature_A", 
    round(abs(random()/92233720368547758.07),0), 
    round(abs(random()/92233720368547758.07),0), 
    round(abs(random()/92233720368547758.07),0),abs(random()%1));"""

# Using round(abs(random()/92233720368547758.07),0) to generate values between 1 and 100
cursor.execute(sql_command) 

sql_command = """INSERT INTO MyDataTable(
    Id, Feature_Name, Feature_Value, Total, N, Percentage) 
    VALUES(abs(random()), "Feature_B", 
    round(abs(random()/92233720368547758.07),0),
    round(abs(random()/92233720368547758.07),0),
    round(abs(random()/92233720368547758.07),0),abs(random()%10));"""
cursor.execute(sql_command)

sql_command = """INSERT INTO MyDataTable(
    Id, Feature_Name, Feature_Value, Total, N, Percentage) 
    VALUES(abs(random()), "Feature_C", 
    round(abs(random()/92233720368547758.07),0),
    round(abs(random()/92233720368547758.07),0),
    round(abs(random()/92233720368547758.07),0),abs(random()) %1);"""
cursor.execute(sql_command)

connection.commit()

connection.close()

connection = sqlite3.connect("resultTable.db")

ResultTable.to_sql("resultTable", connection, if_exists='replace', index = False)


connection.commit()

connection.close()


Image("resultTable.png")

