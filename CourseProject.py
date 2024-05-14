#!/usr/bin/env python
# coding: utf-8

# # Countries and Air Quality
# 
# Author:Seth Abuhamdeh email: seth.abuhamdeh@gmail.com
# 
# Course Project, UC Irvine, Math 10, S23

# ## Introduction
# 
# My project uses a dataset containing thousands of cities and their respective air quality indexes among other measurements relating to air pollution and air quality. In this project, I plan to use machine learning algorithms to experiment and see how well a machine can predict a country based off air quality information. I will compare some machine learning models to see how accurate their predictions are and try to visualize some of their predictions. 

# ## Main Section
# 
# 

# In[ ]:


import pandas as pd
import altair as alt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# ## Cleaning the Data

# In[ ]:


df_pre = pd.read_csv("global air pollution dataset.csv")


# In[ ]:


df_pre = df_pre.dropna()


# In[ ]:


df_pre.shape


# In[ ]:


df_pre


# Since there are 175 countries in this data set, it would be very difficult for any machine to accurately predict a country correctly based off air quality without extreme overfitting. Therefore I will reduce the data set down to the 10 most common countries in the dataset. 

# In[ ]:


top_ten = df_pre["Country"].value_counts().head(10)


# In[ ]:


top_ten = top_ten.index.values


# In[ ]:


top_ten


# In[ ]:


df = df_pre[df_pre["Country"].isin(top_ten)]


# In[ ]:


df


# Now that we have a dataset only containing 10 of the most common countries, it should be easier for the machine to give more accurate predictions of a country based off of its air quality. Since the predictions are for string type data, we should start with a LogisticRegression as our model for our machine.
# 

# In[ ]:


cols = ["AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value" ]


# Since AQI value takes in all values given by CO, Ozone,and NO2 AQI values it is useful to make a bar graph using the average of each country's AQI value to see if there may be some correlation we can draw without the use of a machine. PM2.5 AQI values seem to be equal to overall AQI values so they will be left out.

# In[ ]:


alt.data_transformers.disable_max_rows()
c1 =alt.Chart(df).mark_bar().encode(
    x = "Country",
    y = "mean(AQI Value)",
    color = "Country", 
    tooltip = "mean(AQI Value)"
).properties(
    title = "true data"
)

c2 = alt.Chart(df).mark_point().encode(
    x = "AQI Value",
    y = "CO AQI Value",
    color = "Country",
    tooltip = ["Country", "AQI Value", "CO AQI Value", "City"]  
).properties(
    title = "true data"
)

c3 = c2.mark_point().encode(
    y = "Ozone AQI Value"  
)

c4 = c2.mark_point().encode(
    y = "NO2 AQI Value",  
)

c1|c2|c3|c4


# As we can see, countries like India and China have high AQI values and countries like Russia and Germany have low AQI values. CO AQI values seem to have little effect on overall AQI values, however Ozone has a strong effect on AQI values. NO2 values seems to have a positive correlation as well for most countries except for India, this may be due to bad data or some other unknown variable. Hopefully this will be reflected with our Linear Regression Model. I took the code "c1|c2|c3|c4" from a previous project since I'm not too sure if this was used in class before. Project: https://christopherdavisuci.github.io/UCI-Math-10-S22/Proj/StudentProjects/WenqiZhao.html
# ## Logistic Regression

# I will start off analyzing the data with Logistic Regression since this is the first machine learning model we talked about in class relevant to my data and I beleive will be the least accurate of the models. 

# In[ ]:


reg = LogisticRegression()


# In[ ]:


reg.fit(df[cols], df["Country"])


# In[ ]:


df["logpred"] = reg.predict(df[cols])


# In[ ]:


reg.score(df[cols], df["Country"])


# In[ ]:


reg.coef_


# In[ ]:


reg.classes_


# In[ ]:


reg.intercept_


# In[ ]:


d1 = c1.mark_bar().encode(
     x = "logpred",
    color = "logpred", 
).properties(
    title = "logistic regression"
)
d1|c1


# In[ ]:


d2 = alt.Chart(df).mark_point().encode(
    x = "AQI Value",
    y = "CO AQI Value",
    color = "logpred",
    tooltip = ["Country", "AQI Value", "CO AQI Value", "City","logpred"]  
).properties(
    title = "logistic regression"
)

d3 = d2.mark_point().encode( 
    y = "Ozone AQI Value",
    tooltip = ["Country", "AQI Value", "Ozone AQI Value", "City","logpred"]  ,

)
d4 = d2.mark_point().encode(
    y = "NO2 AQI Value",
    tooltip = ["Country", "AQI Value", "NO2 AQI Value", "City","logpred"]  
)
d2|c2|d3|c3|d4|c4


# Our Logistic Regression model has predicted the right country with close to 50% accuracy which is significantly better than randomly choosing a country(10%). The new graphs show how the linear model predicted the values of certain countries, as we see Mexico's pollution is greatly exagerrated and other countries like India was underestimated.France is not even predicted with this model. We can see a little bit of how the model predicted countries with our point graphs and places where it had inaccuracies. 
# ## Decison Tree Classifier

# Next I'll use a DecisionTreeClassifier since this machine learning model is best model to analyze this type of data that we have used in class.

# In[ ]:


clf1 = DecisionTreeClassifier(max_leaf_nodes= 15)


# In[ ]:


clf1.fit(df[cols], df["Country"])


# In[ ]:


df["DTCpred"] = clf1.predict(df[cols])


# In[ ]:


clf1.score(df[cols], df["Country"])


# In[ ]:


clf2 = DecisionTreeClassifier(max_leaf_nodes= 15)


# Just in case, I will check for overfitting for this model. 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df[cols], df["Country"], train_size = .6, random_state= 18)


# In[ ]:


clf2.fit(X_train, y_train)


# In[ ]:


clf2.score(X_test, y_test)


# Since our DecisonTreeClassifier had very similar accuracy when using real data and train data it is very unlikely that there is any overfitting thus we can continue with this machine model to predict the country from its air pollution values. This model is also about 5% more accurate then the LogisticRegression model.

# In[ ]:


e1 = c1.mark_bar().encode(
     x = "DTCpred",
    color = "DTCpred", 
).properties(
    title = "decision tree classifier"
)
e1|d1|c1


# An interesting point to note is that the DecisionTreeClassifier never predicts France and even Japan yet is still more accurate then the Logistic Regression model. Still this model exaggerates values of those of Mexico and China by a lot.

# In[ ]:


e2 = alt.Chart(df).mark_point().encode(
    x = "AQI Value",
    y = "CO AQI Value",
    color = "DTCpred",
    tooltip = ["Country", "AQI Value", "CO AQI Value", "City","DTCpred"]  
).properties(
    title = "decision tree classifier"
)

e3 = e2.mark_point().encode(
    y = "Ozone AQI Value",
    tooltip = ["Country", "AQI Value", "Ozone AQI Value", "City","DTCpred"]  
)

e4 = e2.mark_point().encode(
    y = "NO2 AQI Value",
    tooltip = ["Country", "AQI Value", "NO2 AQI Value", "City","DTCpred"]  
)
e2|d2|c2|e3|d3|c3|e4|d4|c4


# These graphs depict how the DecisionTreeClassifier predicts countries based off each type of air pollution index. We can see a lot of differences between how clf predicts its countries and how reg predicits its countries. The Decision Tree Classifier seems to section off countries into regions on the graph and whatever point falls into that region is predicted to be that country. This is best seen in the graph for ozone AQI values.

# ## Extra: KNeighborClassifier

# Forthe extra part of this project I am using the KNeighborClassifier as it seems to be another machine learning model that is relevant to my data. 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


reg2 = KNeighborsClassifier(n_neighbors=15)


# In[ ]:


reg2.fit(df[cols],df["Country"])


# In[ ]:


df["KNpred"] = reg2.predict(df[cols])


# Here I use the machine learning algorithm KneighborsClassifier to analyze the data. I get this code from Winter Quarter 2022 Week 6 class notes. https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html

# In[ ]:


reg2.score(df[cols], df["Country"])


# In[ ]:


reg3 = KNeighborsClassifier(n_neighbors= 15)


# In[ ]:


reg3.fit(X_train, y_train)


# In[ ]:


reg3.score(X_test, y_test)


# Our classifier scores somewhat similar on test data and with real data suggesting minimal overfitting thus KNeighborClassifier gives us our most accurate model for this project with nearly 2/3 accuracy much better than random guessing and considerably better than both the DecisionTreeClassifier and LogisticRegressor.

# In[ ]:


f1 = c1.mark_bar().encode(
     x = "KNpred",
    color = "KNpred", 
).properties(
    title = "KNeighbor classifier"
)
f1|e1|d1|c1


# In[ ]:


f2 = alt.Chart(df).mark_point().encode(
    x = "AQI Value",
    y = "CO AQI Value",
    color = "KNpred",
    tooltip = ["Country", "AQI Value", "CO AQI Value", "City","KNpred"]  
).properties(
    title = "KNeighbor classifier"
)

f3 = f2.mark_point().encode(
    y = "Ozone AQI Value",
    tooltip = ["Country", "AQI Value", "Ozone AQI Value", "City","KNpred"]  
)

f4 = f2.mark_point().encode(
    y = "NO2 AQI Value",
    tooltip = ["Country", "AQI Value", "NO2 AQI Value", "City","KNpred"]  
)
f2|e2|d2|c2|f3|e3|d3|c3|f4|e4|d4|c4


# From our graphs, we see that KNeighborClassifier doesn't leave any of the 10 countries out of its predictions and the overall mean of the AQI values for our 3rd prediction column is most similar to those of our original data. We see in our point graphs that KNeighborClassifier has a lot more mixed colors rather than our other models clustering countries together into regions on the graph. KneighborClassifier seems to classify our data much differently than other models which allows it to be more accurate.  

# We see that our machine learning model seems to overpredict AQI values of countries with higher AQI values while underpredicting those with lower AQI values. This is less so with the KNeighborClassifier but still present. 
# ## Visualizing the Predictions

# I finally wanted to visualize the decision boundaries of our different machine learning models. I choose to visualize the ozone AQI values since it seemed to produce the most interesting graph of the 3 values and hopefully will give more conclusive and interesting results of these decision boundaries. All the code for setting up the df_rep dataframe is taken from week 8 monday lecture notes: https://christopherdavisuci.github.io/UCI-Math-10-S23/Week8/Week8-Monday.html

# In[ ]:


rng = np.random.default_rng()


# In[ ]:


arr = rng.random(size = (5000,4))


# In[ ]:


df_rep = pd.DataFrame(arr, columns = cols)


# In[ ]:


df_rep["Ozone AQI Value"] *= 210


# In[ ]:


df_rep["AQI Value"] *= 500


# In[ ]:


df_rep["logpred"] = reg.predict(df_rep[cols])


# In[ ]:


df_rep["DTCpred"] = clf1.predict(df_rep[cols])


# In[ ]:


df_rep["KNpred"] = reg2.predict(df_rep[cols])


# In[ ]:


g1 = alt.Chart(df_rep).mark_point().encode(
    x = "AQI Value",
    y = "Ozone AQI Value",
    color = "logpred",
    tooltip = ["logpred", "Ozone AQI Value", "AQI Value"]

).properties(
    title = "logistic regression"
)
g2 = g1.mark_point().encode(
    color = "DTCpred",
    tooltip = ["DTCpred", "Ozone AQI Value", "AQI Value"]

).properties(
    title = "decision tree classifier"
)
g3 = g1.mark_point().encode(
    color = "KNpred",
    tooltip = ["KNpred", "Ozone AQI Value", "AQI Value"]
).properties(
    title = "KNeighbor classifier"
)
g1|g2|g3


# As we can see, the decision boundaries for the left side of each graph are very different, only Logistic Regression has clear decision boundaries while the DecisionTreeClassifier and KNeighborClassifier has very mixed boundaries. The right side of all 3 graphs pretty much all predict strictly India after around AQI value of 200. While this doesn't completely show how each model makes predictions, it does offer some very interesting insight into how each of these models work and the differences between them. 

# ## Summary
# 
# Either summarize what you did, or summarize the results.  Maybe 3 sentences.

# Over the course of this project we used 3 different machine learning models to analyze air pollution data of 10 countries then use that analysis to predict a country based off of air quality conditions. Our 3 models had surprisingly accurate results with KNeghborClassifier being the most accurate(67%) and Logistic Regression being the least accurate(50%). Each model predicted that Mexico, China, and India were the most polluted countries which is accurate to real data but with some exaggerations especially with China and Mexico. 

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
# 

# * List any other references that you found helpful.

# KNeighborClassifier code: https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html
# some altair code: https://christopherdavisuci.github.io/UCI-Math-10-S22/Proj/StudentProjects/WenqiZhao.html
# Visualizing the Data code:https://christopherdavisuci.github.io/UCI-Math-10-S23/Week8/Week8-Monday.html

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=fa131135-c354-4829-8196-b5850233cffe' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
