# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:13:42 2022

@author: 91995
"""

import pandas as pd
df = pd.read_csv("50_Startups.csv")
df
df.shape
df.dtypes
list(df)

df.duplicated()
df.isnull()
df.isnull().sum()

##################################################################

df["R&D Spend"].hist()
df["Administration"].hist()
df["Marketing Spend"].hist()
df["Profit"].hist()

########################################################################3

t1 = df.groupby("State").size()

t1.plot(kind = "bar")

df.head()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["State"] = LE.fit_transform(df["State"])

########################################################################

X = df.iloc[:,0:4]

Y = df["Profit"]


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)
X_scale
X_scale = pd.DataFrame(X_scale)
X_scale


########################################################################


import matplotlib.pyplot as plt
df.plot.scatter(x =["R&D Spend"],y = ["Administration"] ,color = "black")
df.plot.scatter(x =["R&D Spend"],y = ["Marketing Spend"] ,color = "black")
df.plot.scatter(x =["R&D Spend"],y = ["State"] ,color = "black")
df.plot.scatter(x =["R&D Spend"],y = ["Marketing Spend"] ,color = "black")

###########################################################################################
import seaborn as sns
sns.distplot(df["R&D Spend"])
sns.distplot(df["Administration"])
sns.distplot(df["State"])
sns.distplot(df["Marketing Spend"])


sns.distplot(df["R&D Spend"],kde = False,rug = True)

sns.distplot(df["Administration"],kde = False,rug = True)

sns.distplot(df["State"],kde = False,rug = True)

sns.distplot(df["Marketing Spend"],kde = False,rug = True)

###########################################################################################


sns.jointplot(df["R&D Spend"],df["Administration"],kind = "reg")
sns.jointplot(df["Administration"],df["Marketing Spend"],kind = "reg")
sns.jointplot(df["State"],df["Marketing Spend"],kind = "reg")


sns.jointplot(df["R&D Spend"],df["Administration"],kind = "hex")
sns.jointplot(df["Administration"],df["Marketing Spend"],kind = "hex")
sns.jointplot(df["State"],df["Marketing Spend"],kind = "hex")


###########################################################################################


import numpy as np
Q1 = np.percentile(df["R&D Spend"],25)
Q1
Q2 = np.percentile(df["R&D Spend"],50)
Q3 = np.percentile(df["R&D Spend"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)

df[df["R&D Spend"]<LW]
df[df["R&D Spend"]>UW]

len(df[(df["R&D Spend"]<LW) | (df["R&D Spend"]>UW)])




import numpy as np
Q1 = np.percentile(df["Administration"],25)
Q1
Q2 = np.percentile(df["Administration"],50)
Q3 = np.percentile(df["Administration"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)

df[df["Administration"]<LW]
df[df["Administration"]>UW]

len(df[(df["Administration"]<LW) | (df["Administration"]>UW)])


import numpy as np
Q1 = np.percentile(df["Marketing Spend"],25)
Q1
Q2 = np.percentile(df["Marketing Spend"],50)
Q3 = np.percentile(df["Marketing Spend"],75)
IQR = Q3-Q1
IQR
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)

df[df["Marketing Spend"]<LW]
df[df["Marketing Spend"]>UW]

len(df[(df["Marketing Spend"]<LW) | (df["Marketing Spend"]>UW)])



df.boxplot(column="R&D Spend",vert = False)
df.boxplot(column="Administration",vert = False)
df.boxplot(column="Marketing Spend",vert = False)

##########################################################################################

# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

# B0
LR.intercept_

# B1
LR.coef_

# predictions
Y_pred = LR.predict(X)
Y_pred

# Marics
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y, Y_pred)
print("Mean Squared Error :",mse.round(3))

import numpy as np
print("Root Mean Squared Error :", np.sqrt(mse).round(3))

r2 = r2_score(Y,Y_pred)
print("R square :", r2.round(3))
#===========================================================
import statsmodels.api as sma
Y_new =sma.add_constant(X)
lm2 = sma.OLS(Y,Y_new).fit()
lm2.summary()

##########################################################################################3

RSS =  np.sum((Y_pred-Y)**2)
Y_mean = np.mean(Y)
Y_mean
TSS = np.sum((Y-Y_mean)**2)
R2 = 1-(RSS/TSS)
print("R2:",R2)


vif = 1/(1-R2)
print("VIF value:",vif)


##########################################################################################
import matplotlib.pyplot as plt

import statsmodels.api as sm
qqplot=sm.qqplot(df,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()

list(np.where(df>10))

##########################################################################################


def get_standardized_values(vals):
    return(vals-vals.mean())/vals.std()

plt.scatter(get_standardized_values(df),
            get_standardized_values(df))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()

##########################################################################################














