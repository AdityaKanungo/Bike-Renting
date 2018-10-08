
# coding: utf-8

# # Project Name : Bike Renting
# - The objective of this Case is to Predication of bike rental count on daily based on the environmental and seasonal settings.

# #### Import libraries essential for analysis

# In[598]:


#importing required libraries
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from math import *
from sklearn.metrics import r2_score 
from sklearn.ensemble import ExtraTreesClassifier
#libraries for visualization
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.plotly as py
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[599]:


#set current working directory
os.chdir("C:/Users/BATMAN/Desktop/edWisor Bike Renting")


# #### Load data

# In[600]:


#Load data
df = pd.read_csv("C:/Users/BATMAN/Desktop/edWisor Bike Renting/day.csv")


# In[601]:


df.head(15)


# In[602]:


df.describe()


# In[603]:


df.shape


# In[604]:


df.nunique()


# In[605]:


#storing col-names of category and numeric variables
num_var = ['temp','atemp','hum','windspeed','cnt']
cat_var = ['season','yr','mnth','holiday','weekday','workingday','weathersit']

##Converting datatype to category
#for i in cat_var:
#    df[i] = df[i].astype('category')


# #### Missing Value Analysis

# In[606]:


#Missing value analysis
print(df.isnull().sum())


# - Data does not have any missing values.

# #### Outlier Analysis

# In[607]:


#Outlier detection using box-plot method
import seaborn as sns
y0 = df['temp']
y1 = df['atemp']
y2 = df['hum']
y3 = df['windspeed']

df_num = df.drop(['instant','dteday','casual','registered','holiday','yr','mnth','weekday','workingday','season','weathersit','cnt'], axis=1)
ax = sns.boxplot(data=df_num, orient="h", palette="Set2")


# - From above box-plot we can see that 'hum' and 'winspeed' have few outliers.

# In[608]:


#Fiding outliers and replacing with NA
for i in num_var:
    for j in range(len(df)):
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        
        IQR = Q3 - Q1
        if (df[i].iloc[j] <= (Q1 - 1.5*IQR) or df[i].iloc[j] >= (Q3 + 1.5*IQR)):
            df[i].iloc[j] = np.nan


# In[609]:


#Count the number of outliers in 'hum' and 'windspeed'
print(df['hum'].isnull().sum())
print(df['windspeed'].isnull().sum())


# In[610]:


#imputing outliers with average value of the variable
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['windspeed'] = df['windspeed'].fillna(df['windspeed'].mean())


# #### Exploratory Data Analysis (EDA)

# #### Atrribute information : 
# - Instant : Record index
# - dteday : date
# - season : Date
# - season: Season (1:springer, 2:summer, 3:fall, 4:winter)
# - yr: Year (0: 2011, 1:2012)
# - mnth: Month (1 to 12)
# - holiday: weather day is holiday or not (extracted fromHoliday Schedule)
# - weekday: Day of the week (Assuming 0 as sunday and 1 as monday and so on)
# - workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
# - weathersit: (extracted fromFreemeteo)
#     1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp: Normalized temperature in Celsius. The values are derived via
#     (t-t_min)/(t_max-t_min),t_min=-8, t_max=+39 (only in hourly scale)
# - atemp: Normalized feeling temperature in Celsius. The values are derived via
#     (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# - hum: Normalized humidity. The values are divided to 100 (max)
# - windspeed: Normalized wind speed. The values are divided to 67 (max)
# - casual: count of casual users
# - registered: count of registered users
# - cnt: count of total rental bikes including both casual and registered (TARGET VARIABLE)

# #### Key observations about dataset  : 
# 
# - Normalized variables : 'temp','atemp','hum','windspeed'.
# - Sum of 'casual' and 'registered' is equal to target variable i.e 'cnt'.
# - instant and date are variables with all unique values.
# - Numeric variables : 'instant','temp','atemp','hum','windspeed','casual','registered','cnt'.
# - Catergory variables : 'season','yr','mnth','holiday','weekday','workingday','weathersit'.
# - date variable  : dteday

# #### Converting normalized features back to un-scaled form for better visualization and interpretation.
# 
# #### Given :
# 
# - temp: Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in  hourly scale)
# - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max- t_min), t_min=-16, t_max=+50 (only in hourly scale) 
# - hum: Normalized humidity. The values are divided to 100 (max)
# - windspeed: Normalized wind speed. The values are divided to 67 (max)

# In[611]:


#Converting the normalized value to real value
df['temp'] = ((df['temp']*47)-8)
df['atemp'] = ((df['atemp']*66)-16)
df['hum'] = df['hum']*100
df['windspeed'] = df['windspeed']*67


# In[612]:


#Disturbutions of required numerical variables
# histogram plot
sns.distplot(df['cnt']);
pyplot.show()
#temp
sns.distplot(df['temp']);
pyplot.show()
#atemp
sns.distplot(df['atemp']);
pyplot.show()
#humidity
sns.distplot(df['hum']);
pyplot.show()
#windspeed
sns.distplot(df['windspeed']);
pyplot.show()


# In[613]:


import seaborn as sns
import matplotlib.pyplot as plt
#Season
#plot1
sns.set(style="ticks", color_codes=True)
sns.catplot(x='season', y='cnt', kind="box", data=df);
#plot-2
sns.set(style="whitegrid")
plt.figure(figsize=(6,6))
ax = sns.barplot(x="season", y="cnt", data=df)
#plot-3
sns.catplot(x="weathersit", y="cnt", hue="season", kind="bar", data=df);

#Here Season - > (1:springer, 2:summer, 3:fall, 4:winter)


# ##### from above plots of  we observed that:
# - bike renting count was maximum in fall and summer
# - & least in springer

# In[614]:


#Year
#plot-1
sns.set(style="whitegrid")
plt.figure(figsize=(4,5))
ax = sns.barplot(x="yr", y="cnt", data=df)
#plot-2
sns.catplot(x="yr", y="cnt", hue="season", kind="bar", data=df);

#Here year - > (0:2011, 1:2012)


# ##### from above plots of  year we observed that:
# - bike renting count was more in 2012 as compared to 2011

# In[615]:


#Month
#plot-1
sns.set(style="whitegrid")
plt.figure(figsize=(10,4))
ax = sns.barplot(x="mnth", y="cnt", data=df)
#plot-2
sns.catplot(x="holiday", y="cnt", hue="season", kind="bar", data=df);
#plot-2
sns.set(style="ticks", color_codes=True)
sns.catplot(x='mnth', y='cnt', kind="box", data=df);

#Here : Month (1 to 12) are (jan-Dec)


# ##### from above plots of months we observed that:
# - bike renting count was more in summer and fall months as compared to winter months

# In[616]:


#holiday
#plot-1
sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
ax = sns.barplot(x="holiday", y="cnt", data=df)
#plot-2
sns.catplot(x="holiday", y="cnt", hue="season", kind="bar", data=df);


# ##### from above plots of holiday we observed that:
# - there is no strong relation between holiday and bike count

# In[617]:


#Weekday
#plot-1
sns.set(style="ticks", color_codes=True)
sns.catplot(x='weekday', y='cnt', kind="box", data=df);
#plot-2
sns.set(style="whitegrid")
plt.figure(figsize=(10,4))
ax = sns.barplot(x="weekday", y="cnt", data=df)
#plot-3
g = sns.FacetGrid(df, hue="holiday", palette="Set1", height=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "weekday", "registered", s=100, linewidth=.5, edgecolor="white")
g.add_legend();

#Here ->  Month (0:6) are (Sun:Sat))


# ##### from above plots of weekday we observed that:
# - there is no strong relation between weekday and bike count

# In[618]:


#Workingday
#plot-1
sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
#plot-2
ax = sns.barplot(x="workingday", y="cnt", data=df)
sns.catplot(x="workingday", y="cnt", hue="season", kind="bar", data=df);


# ##### from above plots of workingday we observed that:
# - there is no strong relation between workingday and bike count

# In[619]:


#Weathersit
#plot-1
sns.set(style="ticks", color_codes=True)
sns.catplot(x='weathersit', y='cnt', kind="box", data=df);
#plot-2
sns.set(style="whitegrid")
plt.figure(figsize=(4,5))
ax = sns.barplot(x="weathersit", y="cnt", data=df)

#Here ->  weather (1 to 4) are 
#   - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
#   - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#   - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#   - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog


# ##### from above plots of weather we observed that:
# - Maximun bike rental counts are in weather 1 : Clear, Few clouds, Partly cloudy, Partly cloudy
# - Minimun bike rental counts are in weather 3 : Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - Medium bike rental counts are in weather 2 : Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - Zero bike rental in weather 4 : Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

# In[620]:


#temp-cnt plot season-wise
sns.set(style="ticks")
g = sns.FacetGrid(df, col="season", hue="yr")
g.map(plt.scatter, "temp", "cnt", alpha=.7)
g.add_legend();


# ##### from above plots of temp we observed that:
# - there is mild relation between weekday and bike count

# In[621]:


#atemp-cnt plot season-wise
sns.set(style="ticks")
g = sns.FacetGrid(df, col="season", hue="yr")
g.map(plt.scatter, "atemp", "cnt", alpha=.7)
g.add_legend();


# ##### from above plots of atemp we observed that:
# - there is mild relation between atemp and bike count

# In[622]:


#hum-cnt plot season-wise
sns.set(style="ticks")
g = sns.FacetGrid(df, col="season", hue="yr")
g.map(plt.scatter, "hum", "cnt", alpha=.7)
g.add_legend();


# ##### from above plots of humidity we observed that:
# - there is no strong relation between humidity and bike count

# In[623]:


#windspeed-cnt plot season-wise
sns.set(style="ticks")
g = sns.FacetGrid(df, col="season", hue="yr")
g.map(plt.scatter, "windspeed", "cnt", alpha=.7)
g.add_legend();


# ##### from above plots of windspeed we observed that:
# - there is no strong relation between windspeed and bike count

# In[624]:


#Normalization
df['temp'] = (df['temp'] - min(df['temp']))/ (max(df['temp']) - min(df['temp']))
df['atemp'] = (df['atemp'] - min(df['atemp']))/ (max(df['atemp']) - min(df['atemp']))
df['hum'] = (df['hum'] - min(df['hum']))/ (max(df['hum'])- min(df['hum']))
df['windspeed'] = (df['windspeed'] - min(df['windspeed']))/ (max(df['windspeed']) - min(df['windspeed']))


# #### Feature Selection

# In[625]:


#Correlation Analysis using heatmap

df_corr = df.iloc[:,9:13]
f, ax = plt.subplots(figsize=(10,10))
plt.title('Correlation between numerical predictors',size=14,y=1.05)
corr = df_corr.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap= sns.diverging_palette(220,10, as_cmap = True), square=True,
            annot = True,ax=ax)


# - The above heatmap indicates high positive correlation between temp and a temp.
# - So, we need to drop atemp to avoid feeding redundant information to our model.

# In[626]:


# Dropping variables unessential for analysis.
df = df.drop(['dteday','casual','mnth','holiday'], axis=1)


# #### Reason for dropping variables/features:
# 
# - atemp: highly +vely correlated to temp
# - instant: unique value, might lead to overfitting
# - dteday: information already stored in month and year
# - casual and registered : sum of these two variables is equal to the target variable

# In[627]:


df.dtypes


# #### Sampling

# In[628]:


def mape(y_true, y_pred):
    mape = np.mean(np.absolute((y_true - y_pred) / y_true))*100
    return mape
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[629]:


#Splitting the data into train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)


# #### Modeling

# In[630]:


X_train = train.drop(['cnt'], axis=1)
Y_train = train["cnt"]
X_test  = test.drop(['cnt'], axis=1)
Y_test = test['cnt']
X_train.shape, Y_train.shape, X_test.shape,Y_test.shape


# In[631]:


#Randomforest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 45)
rf.fit(X_train,Y_train)

train_prediction = rf.predict(X_train)
test_prediction = rf.predict(X_test)

print('test ->')
print("R^2  : ",r2_score(Y_test,test_prediction))
print("MAPE :",mape(Y_test,test_prediction))
print("RMSE :",rmse(Y_test,test_prediction))
print('train ->')
print("R^2  : ",r2_score(Y_train,train_prediction))
print("MAPE :",mape(Y_train,train_prediction))
print("RMSE :",rmse(Y_train,train_prediction))


# In[632]:


#Decision tree
from sklearn.tree import DecisionTreeRegressor

fit = DecisionTreeRegressor(max_depth=5).fit(X_train,Y_train)

train_prediction = fit.predict(X_train)
test_prediction = fit.predict(X_test)

print('test ->')
print("R^2  : ",r2_score(Y_test,test_prediction))
print("MAPE :",mape(Y_test,test_prediction))
print("RMSE :",rmse(Y_test,test_prediction))
print('train ->')
print("R^2  : ",r2_score(Y_train,train_prediction))
print("MAPE :",mape(Y_train,train_prediction))
print("RMSE :",rmse(Y_train,train_prediction))


# In[633]:


#Liner Regression
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)

train_prediction = regr.predict(X_train)
test_prediction = regr.predict(X_test)

print('test ->')
print("R^2  : ",r2_score(Y_test,test_prediction))
print("MAPE :",mape(Y_test,test_prediction))
print("RMSE :",rmse(Y_test,test_prediction))
print('train ->')
print("R^2  : ",r2_score(Y_train,train_prediction))
print("MAPE :",mape(Y_train,train_prediction))
print("RMSE :",rmse(Y_train,train_prediction))

