#!/usr/bin/env python
# coding: utf-8

# This project is about a team of plantation planners are concerned about the yield of oil palm trees, which seems to fluctuate. They have collected a set of data and needed help in analysing on how external factors influence fresh fruit bunch (FFB) yield. Thus, this work will cover analysis of important factors that influence FFB using Ordinary Least Square (OLS) technique.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from scipy import stats


# Initially, the necessary packages are imported for mathematical calculations, dataframe operations, visualizations and statistical equations.

# In[2]:


df=pd.read_csv('palm_ffb.csv')


# Next, the provided dataset on FFB yield and its identified potential factors imported to the Jupyter Notebook.

# In[3]:


df.head(20)


# In[4]:


df.shape


# In[5]:


df.info()


# Here, it can be observed that the dataset is a time series, with Month as its frequency. Overall, the dataset consists of 9 columns and 130 rows, where all of the columns except Date is in the form of float and integer data type.

# In[6]:


df.describe()


# The dataset consists of readings collected for 130 months. During that time, average FFB yield recorded is 1.60, where the minimum and maximum range from 1.08 to 2.27. Thus, the standard deviation is 0.28. In terms of working days, on average, 25 days allocated for taking care of the plantation. Average ambient temperature, Soil moisture and precipitation of rain are 26.85, 527.65 and 188.98 respectively.

# In[7]:


df.isnull().sum()


# The dataset is complete, where there is no null values. 

# In[8]:


df.columns


# In[9]:


df['Working_days'].unique()


# In[10]:


col=['SoilMoisture', 'Average_Temp', 'Min_Temp', 'Max_Temp',
       'Precipitation', 'Working_days', 'HA_Harvested', 'FFB_Yield']

for i in col:
    stat, p = stats.normaltest(df[i])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print(i,' looks Gaussian (fail to reject H0)')
        print(' ')
    else:
        print(i,' does not look Gaussian (reject H0)')
        print(' ')


# Normality of distributions of each attributes of this dataset being tested. This is because we are using OLS for the analysis, all of the attributes must be normally ditributed (Gaussian). Normal distribution means the attribute is linear fitted, which is a must for OLS. Using Shapiro-Wilk test, it is discovered that all of the attributes are normally distributed, except Min_Temp and Working_days. However, for ambient temperature, only Average_Temp will be used as Min_Temp, Precipitation and Max_Temp not representative of real ambient temperature where the plantations are growing. Next, the rain Precipitation need to be visualized further to understand the distributions. Lastly, the Working_days are proved to be categorical data, based on df['Working_days'].unique(). Thus, this attribute must be encoded first before proceed to OLS.

# In[11]:


def diagnostic_plots(df, variable):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()


# In[12]:


col=['SoilMoisture', 'Average_Temp', 'Min_Temp', 'Max_Temp',
       'Precipitation', 'Working_days', 'HA_Harvested', 'FFB_Yield']

for i in col:
    diagnostic_plots(df, i)


# Based on Shapiro-Wilk test previously, histogram, probability plot and boxplot of each attribute visualized, to firm up our understanding on the distribution patterns. Focusing on the non-normally distributed attributes, for Min_temp and Precipitation, it is discovered that the non-normality being contributed by the outliers. Thus, in order to ensure the OLS is not misleading, the outliers must be capped using Winsorization technique. With this transformation, the linearity and distribution can be improved.

# In[13]:


df2=df.loc[:,['SoilMoisture', 'Average_Temp','Precipitation', 'Working_days', 'HA_Harvested', 'FFB_Yield']]


# To proceed, the OLS model will be developed by using SoilMoisture, Average_Temp,Precipitation, Working_days, HA_Harvested, FFB_Yield as discussed previously. Date will be omitted since the analysis is only to find significant factor, not to do time series forecasting.

# In[14]:


# calculate the interquantile range

for column in df2:
  IQR = df2[column].quantile(0.75) - df2[column].quantile(0.25)
  lower_boundary = df2[column].quantile(0.25) - (IQR*1.5)
  upper_boundary = df2[column].quantile(0.75) + (IQR*1.5)
  print('IQR for {} = {} - {}'.format(column,lower_boundary,upper_boundary))


# In[15]:


df2.columns


# In[16]:


get_ipython().system('pip install feature_engine')
from feature_engine.outliers import Winsorizer

windsoriser = Winsorizer(capping_method='iqr', # choose iqr for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['SoilMoisture', 'Average_Temp', 'Precipitation', 'Working_days',
                                     'HA_Harvested', 'FFB_Yield'])
windsoriser.fit(df2)


# In[17]:


df3 = windsoriser.transform(df2)


# To ensure the normal distribution and linearity of each attribute, Winsorization technique is implemented. Here, the value of upper boundary and lower boundary of each attribute is capped. Meaning that, all observations greater than the 75th percentile equal to the value at the 75th percentile and all observations less than the 25th percentile equal to the value at the 25th percentile. With this technique, extreme values are rectified within the specified percentile. 

# In[18]:


# get the dummies and store it in a variable
dummies = pd.get_dummies(df['Working_days'])


# In[19]:


# Concatenate the dummies to original dataframe
df4 = pd.concat([df3, dummies], axis='columns')


# In[20]:


# drop the values
df4=df4.drop(['Working_days'], axis='columns')


# In[21]:


df4.head()


# In[22]:


df4.set_axis(['SoilMoisture','Average_Temp','Precipitation',
              'HA_Harvested','FFB_Yield','21_working_days','22_working_days','23_working_days',
              '24_working_days','25_working_days','26_working_days','27_working_days'],axis=1,inplace=True)


# In[23]:


df4=df4.loc[:,['SoilMoisture','Average_Temp','Precipitation',
              'HA_Harvested','FFB_Yield','21_working_days','22_working_days','23_working_days',
              '24_working_days','25_working_days','26_working_days','27_working_days']]


# In[24]:


df4.head()


# Next, the Working_days attribute is pre-processed. Here, this attribute, which categorical data is encoded using dummies, which consists of 0 and 1, as a code to represent each category; 21_working_days,22_working_days,23_working_days,            24_working_days,25_working_days,26_working_days, and 27_working_days. With this attribute encoded, this attributes can be utilized in the OLS.

# In[25]:


df4.describe()


# In[26]:


df4.shape


# The descriptive statistics of the dataset being observed again. Here, the minimum and maximum values of each attribute has been updated based on Winsorization. New columns are added; the encoded categorical variable, Working_days is divided into new 6 columns; 21_working_days,22_working_days,23_working_days, 24_working_days,25_working_days,26_working_days, and 27_working_days, based on unique categories identified. Thus, in the updated dataset, it consists of 12 columns and 130 rows.

# In[27]:


df4.columns


# In[28]:


df4['21_working_days'] = df4['21_working_days'].astype(float)
df4['22_working_days'] = df4['22_working_days'].astype(float)
df4['23_working_days'] = df4['23_working_days'].astype(float)
df4['24_working_days'] = df4['24_working_days'].astype(float)
df4['25_working_days'] = df4['25_working_days'].astype(float)
df4['26_working_days'] = df4['26_working_days'].astype(float)
df4['27_working_days'] = df4['27_working_days'].astype(float)


# In[29]:


df4.info()


# In[30]:


col=['SoilMoisture', 'Average_Temp',
       'Precipitation', 'HA_Harvested', 'FFB_Yield', '21_working_days',
       '22_working_days', '23_working_days', '24_working_days',
       '25_working_days', '26_working_days', '27_working_days']

for i in col:
    stat, p = stats.normaltest(df4[i])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print(i,' looks Gaussian (fail to reject H0)')
        print(' ')
    else:
        print(i,' does not look Gaussian (reject H0)')
        print(' ')


# In[31]:


col=['SoilMoisture', 'Average_Temp',
       'Precipitation', 'HA_Harvested', 'FFB_Yield', '21_working_days',
       '22_working_days', '23_working_days', '24_working_days',
       '25_working_days', '26_working_days', '27_working_days']

for i in col:
    diagnostic_plots(df4, i)


# Shapiro-Wilk test for normality testing being run again. This time, Precipitation attribute is normally distributed as the extreme values has been capped by Winsorization technique. However, the Working_days attribute is not in normal distribution as the attribute only being encoded for each categories. Then, the distribution being visualized for better understanding.

# In[32]:


X, y= df4[['SoilMoisture', 'Average_Temp','Precipitation', 'HA_Harvested', '21_working_days',
       '22_working_days', '23_working_days', '24_working_days','25_working_days', '26_working_days', 
           '27_working_days']], df4[['FFB_Yield']]


# In[33]:


X = sm.add_constant(X)


# In[34]:


model = sm.OLS(y, X).fit()


# In[35]:


print(model.summary())


# Finally, the OLS model being developed, using statsmodels.api library. Here, it is observed that the p-values of SoilMoisture, Precipitation, HA_Harvested and Working_days are below 0.05. Meaning that, these are the significant factors which influenced the FFB yield. However, in this model, Average_Temp discovered to be not significant. In order to ensure the accurateness of this statistical model, we cross-check the results with technical papers from this domain. Based on Second National Communication (2011) by Ministry of Natural Resources Environment (NRE), Putrajaya, Malaysia, optimum temperature
# for oil palm production is between 22°C and 32°C with evenly distributed rainfall at an annual mean between 2000 mm and 3000
# mm. Which means, the importance of ambient temperature cannot be ignored. In addition, Oil palm yield can drop by approximately
# 30%, when the temperature increases by 2°C above the optimum levels and when rainfall decreases by 10%. In addition, heavy rainfall can decrease FFB yields, due to the excess moisture stress experienced by the oil palms (Goh et al., 2002). In terms of Working_days, this factor can be discussed in the perspective of taking care the crops, where the activities including weeding, fertilizing, pruning and harvesting. According to Idris et al. (2006), low fertiliser rate is the main factor that affects FFB yield, which means that low working days will impact the FFB yield negatively.

# References :
# 
# 1) Second National Communication (2011). The Report Submitted to United Nations Framework Convention on Climate Change (UNFCCC). Ministry of Natural Resources Environment (NRE), Putrajaya, Malaysia.
# 2) Goh, K J; Kee, K K; Chew, P S; Gan, H H; Heng, Y C and Ng, H C P (2002). Concept of site yield potential and its applications in oil palm plantations. Malaysian Oil Sci. Technol., 11: 57-63.
# 3) Idris, O; Amran, A; Nek, D A and Mohd, H J (2006). Impak Pemindahan Teknologi Pusat TUNAS di Sabah ke Atas Peserta Ladang Pekebun Kecil Sawit. MPOB, Bangi.
# 4) Sahidan, A. S. (2021, September 30). FACTORS AFFECTING FRESH FRUIT BUNCH YIELDS OF INDEPENDENT SMALLHOLDERS IN SABAH. Oil Palm Industry Economic Journal, 21(2), 22–34. 
