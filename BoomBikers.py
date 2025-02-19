#!/usr/bin/env python
# coding: utf-8

# # ***Simple Linear Regression***
# 
# In this notebook, we'll build a linear regression model to predict boom-bikes sales using an appropriate predictor variable.

# ## Step 1: Reading and Understanding the Data

# #### Import Libraries

# In[179]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# #### Load the dataset

# In[180]:


# Load the dataset
df = pd.read_csv("C:/Users/mahes/Downloads/download.csv") 


# #### Print out to check if we get the data

# In[181]:


print(df)


# #### Display information and check for Missing Values

# In[182]:


# Display basic info
print(df.info())


# In[183]:


print(df.describe())


# In[184]:


# Checking for missing values
print(df.isnull().sum())


# ## Step 2: Visualising the Data

# Visualise our data using seaborn. We'll first make a pairplot of all the variables present to visualise which variables are most correlated to `CNT`.

# In[185]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[186]:


sns.pairplot(df, x_vars=['hum','temp','windspeed'], y_vars='cnt',size=4, aspect=1, kind='scatter')
plt.show()


# In[187]:


plt.figure(figsize=(10, 16))
sns.heatmap(df.drop(columns = 'dteday').corr(), cmap="YlGnBu", annot = True)
plt.show()


# As visible from the pairplot and the heatmap, the variable 'registered' seems to be most correlated with 'cnt'. We have the option to use only the variables with the high correlation values ,however, we'll go ahead and perform simple linear regression using most of the variables.

# ------
# # Step 3: Perform Simple Linear Regression

# 
# ## Summary
# 
# Equation of linear regression<br>
# $y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$
# 
# -  $y$ is the response
# -  $c$ is the intercept
# -  $m_1$ is the coefficient for the first feature
# -  $m_n$ is the coefficient for the nth feature<br>
# 
# In our case:
# 
# $y = c + m_1 \times TV$
# 
# The $m$ values are called the model **coefficients** or **model parameters**.
# 
# ---

# ## Data Cleaning and Pre-processing

# ### Convert categorical variables

# In[188]:


# Convert categorical variables to appropriate types
df['season'] = df['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
df['weathersit'] = df['weathersit'].map({1: 'clear', 2: 'mist', 3: 'light_snow', 4: 'heavy_rain'})
df['yr'] = df['yr'].map({0: 2018, 1: 2019})


# In[189]:


# Drop column dteday or any columns which are not required
df = df.drop(columns = 'dteday')


# In[190]:


df = pd.get_dummies(df, drop_first=True)  # Creating dummy variables


# ## Split data into Independent and Dependent Variables
# 
# We first assign the feature variables to the variable `X` and the response variable, `cnt`, to the variable `y`.

# In[191]:


# Define independent and dependent variables
X = df.drop(columns=['cnt', 'instant', 'casual', 'registered'])
y = df['cnt']


# ### Split data - Training and testing datasets

# In[192]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8,test_size=0.2, random_state=100)


# In[193]:


# Let's now take a look at the train dataset

print(X_train.head())


# In[194]:


print(y_train.head())


# ### Standardization
# 
# StandardScaler() is a preprocessing technique in scikit-learn used for standardizing features by removing the mean and scaling to unit variance. StandardScaler operates on the principle of normalization, where it transforms the distribution of each feature to have a mean of zero and a standard deviation of one. This process ensures that all features are on the same scale, preventing any single feature from dominating the learning process due to its larger magnitude.

# In[195]:


# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ### Building a Linear Model
# 
# The `statsmodels` library, by default, fits a line on the dataset which passes through the origin. But in order to have an intercept, you need to manually use the `add_constant` attribute of `statsmodels`. And once you've added the constant to your `X_train` dataset, you can go ahead and fit a regression line using the `OLS` (Ordinary Least Squares) attribute of `statsmodels` as shown below

# In[196]:


# Add constant term for OLS model
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)


# In[197]:


# Fit the regression line using OLS
lin_reg = sm.OLS(y_train, X_train_scaled).fit()


# In[198]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
print(lin_reg.params)


# In[199]:


print(lin_reg.summary())


# ####  Looking at some key statistics from the summary
# 
# The values we are concerned with are - 
# 1. The coefficients and significance (p-values)
# 2. R-squared
# 3. F statistic and its significance

# In[200]:


# Extract significant variables
significant_vars = lin_reg.pvalues[lin_reg.pvalues < 0.05]
print("Significant Variables:")
print(significant_vars)


# Significant variable taken as `windspeed`.

# ##### 1. The coefficient for Windspeed is -254.26, with a very low p value
# The coefficient is statistically significant. So the association is not purely by chance. 

# ##### 2. R - squared is 0.84
# Meaning that 84.0% of the variance in `CNT` is explained by `Windspeed`

# ###### 3. F statistic has a very low p value (practically low)
# Meaning that the model fit is statistically significant, and the explained variance isn't purely by chance.

# ---
# The fit is significant. Let's visualize how well the model fit the data.
# 
# From the parameters that we get, our linear regression equation becomes:
# 
# $ CNT = 4505.2671 + -254.2619 \times Windspeed $

# In[201]:


plt.scatter(X_train['windspeed'], y_train)
plt.plot(X_train['windspeed'], 4505.2671 + -254.2619*X_train['windspeed'], 'r')
plt.show()


# ## Step 4: Residual analysis 
# To validate assumptions of the model, and hence the reliability for inference

# #### Distribution of the error terms
# We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[202]:


y_train_pred = lin_reg.predict(X_train_scaled)
res = (y_train - y_train_pred)


# In[203]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# The residuals are following the normally distributed with a mean 0. All good!

# #### Looking for patterns in the residuals

# In[204]:


plt.scatter(X_train['windspeed'],res)
plt.show()


# We are confident that the model fit isn't by chance, and has decent predictive power. The normality of residual terms allows some inference on the coefficients.
# 
# Although, the variance of residuals increasing with X indicates that there is significant variation that this model is unable to explain.

# ## Step 5: Predictions on the test set

# In[205]:


# Make predictions

# Add a constant to X_test
#X_test_sm = sm.add_constant(X_test['windspeed'])

#X_test_sm.reshape(-1)
y_pred = lin_reg.predict(X_test_scaled)


# In[206]:


print(y_pred)


# In[207]:


# Evaluate the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ##### Looking at the RMSE

# In[208]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# ###### Checking the R-squared on the test set

# In[209]:


r2 = r2_score(y_test, y_pred)
print(f'R-squared score on test set: {r2:.4f}')


# ##### Visualizing the fit on the test set

# In[210]:


plt.scatter(X_test['windspeed'], y_test)
plt.plot(X_test['windspeed'], 4505.2671 + -254.2619*X_test['windspeed'], 'r')
plt.show()

