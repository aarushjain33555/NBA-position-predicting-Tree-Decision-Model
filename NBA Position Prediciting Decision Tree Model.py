#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from pandas_ods_reader import read_ods
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
#importing all necessary libraries needed for the model to run
path = '/home/aarush/Documents/NBAStats.ods'
sheet_index = 1
sheet_name = "Sheet1"
df1 = read_ods(path, sheet_index)
#converts the ods file to a dataframe
df1 = df1.drop([54])
df1 = df1[df1.PLAYER != "PLAYER"]
#filters out the rows that have no actual player/stats on the list

df1 = df1.drop(columns = ["RK"])
#drops the column with the header 'RK'

df1 = df1.reset_index().drop(columns=["index"])
#drops the column with the indexes

dataset = df1


x_PPG = df1.PTS
df1[["PTS"]]
df1 = df1.drop(columns = ["PTS"])
x_BPG = df1.BLKPG
df1[["BLKPG"]]
df1 = df1.drop(columns = ["BLKPG"])
x_APG = df1.APG
df1[["APG"]]
df1 = df1.drop(columns = ["APG"])
#seperates the columns with points, assists, and blocks from the data into mini dataframes 

x_values = {
    "PPG" : x_PPG,
    "APG" : x_APG,
    "BPG" : x_BPG
}
#creates a dictionary that temporarily stores the values for x

final_x_values = pd.DataFrame(x_values)
#converts x values into one huge dataframe
df3 = []
def parse_position(row_element):
    #print(row_element["PLAYER"][-1])
    if row_element["PLAYER"][-1] == "G": 
        df3.append("G")
             
    elif row_element["PLAYER"][-1] == "F":
        
        df3.append("F")
            
    else: 
        df3.append("C")
#a function that will be used later to check the y values which is the positions
df1 = df1.apply(lambda x: parse_position(x), axis=1)
#applies previous function through a lambda function to scroll through columns and collect y values


y_values = {
    "Position" : df3
}
#converts the y values into a dictionary

final_y_values = pd.DataFrame(y_values)
#converts the dictionary into a data frame 

X_train, X_test, Y_train, Y_test = train_test_split(final_x_values, final_y_values, test_size = 0.20)

data_trainer = DecisionTreeClassifier()
data_trainer.fit(X_train, Y_train)
#trains the model with 80 percent of the data with both x and y values


y_predictor = data_trainer.predict(X_test)
#predicts the y values with the remaining x values that make up 20 percent of data

print(confusion_matrix(Y_test, y_predictor))
print("MODEL REPORT")
print(classification_report(Y_test, y_predictor))
#prints the results and accuracy of the model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




