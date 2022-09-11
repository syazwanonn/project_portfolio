#!/usr/bin/env python
# coding: utf-8

# This work is about an analysis of petrol formulations. A customer informed their consultant that they have developed several formulations that gives different characteristics of burning pattern. The formulations are obtaining by adding varying levels of additives that, for example, prevent engine knocking, gum prevention, stability in storage, and etc. Here, the consultant will be assisted on establishing descriptive statistics, data visualization and clustering of formulations.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[2]:


df=pd.read_csv('ingredient.csv')


# ingredient dataset is used in this analysis, where ingredients of additives namely a to i are varied for each formulation. The formulations are obtaining by adding varying levels of additives that, for example, prevent engine knocking, gum prevention, stability in storage, and etc. In order to confirm the characteristics of each ingredient, statistical analysis carried out.

# In[3]:


df


# Here, 214 formulations were developed using said additives.

# In[4]:


df.describe()


# In this formulation, it is discovered that, amount of each additive added in average are 1.518365, 13.408, 2.684, 1.444, 72.651, 0.497, 8.957, 0.175 and 0.057 for a until i respectively. For the quantity added, additive c has largest standard deviation, while a has the lowest. Additive e take up the most quantity compare to others, where the quantity used up to 75.41.

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# Overall, the data types of this dataset are in float type, and there is no missing values detected.

# In[7]:


col = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

for i in col:
    stat, p = stats.normaltest(df[i])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print(i,' looks Gaussian (fail to reject H0)')
    else:
        print(i,' does not look Gaussian (reject H0)')
        print(' ')


# Normality test aka Shapiro Wilk test being used to identify whether each additive distribution is Gaussian or not. Based on the statistics test, null hypothesis is the distribution is Gaussian, while the alternative hypothesis is the distribution is not Gaussian. Thus, in this test, all of the additives not distributed in Gaussian manner.

# In[8]:


col = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

for i in col:
    df[i].hist(bins=20)
    plt.title(i)
    plt.show()
    print( 'Kurtosis of normal distribution: {}'.format(stats.kurtosis(df[i])))
    print( 'Skewness of normal distribution: {}'.format(stats.skew(df[i])))


# Visualization using histogram being done to confirm the distribution pattern. Here, kurtosis and skewness also calculated. It is discovered that additive f is the most heavily skewed compare to the others.

# In[9]:


sns.pairplot(df)


# In[10]:


plt.figure(figsize=(10,10))
corr=df.corr(method='spearman')
matrix = np.triu(corr)
sns.heatmap(corr,cmap='GnBu',annot=True,mask=matrix)


# Pairwise graph is plotted for the overall dataset. Here, the correlation between each additive visualized. It can be observed that, the correlations between additives are weak. Thus, it can be said that the additives are independent and no strong relationship detected. Meaning that, the increasing and decreasing of one additive does not influenced the amount of other corresponding additives.

# In[11]:


df.boxplot(figsize=(15,10))
plt.title('Boxplot of additive types')
plt.xlabel('Type')
plt.ylabel('Level')


# Each additives assumed to be independent of each other. Also, all of them are non-Gaussian distributed. Meaning that, non-parametric test will be used. To test the distribution difference between two samples, Mann-Whitney U test is used. The hypothesis of this test as per shown below:
# 
# Fail to Reject H0: Sample distributions are equal.
# Reject H0: Sample distributions are not equal.

# In[12]:


stat, p = stats.mannwhitneyu(df['c'], df['d'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# Since the p-value (0.00) is less than 0.05, we reject the null hypothesis. We have sufficient evidence to say that the true mean level used for the formulation is different between the two groups.

# In[13]:


stat, p = stats.mannwhitneyu(df['d'], df['e'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# Since the p-value (0.00) is less than 0.05, we reject the null hypothesis. We have sufficient evidence to say that the true mean level used for the formulation is different between the two groups.

# In[14]:


stat, p = stats.mannwhitneyu(df['d'], df['f'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# Since the p-value (0.00) is less than 0.05, we reject the null hypothesis. We have sufficient evidence to say that the true mean level used for the formulation is different between the two groups.

# In[15]:


stat, p = stats.mannwhitneyu(df['h'], df['i'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# Since the p-value (0.00) is less than 0.05, we reject the null hypothesis. We have sufficient evidence to say that the true mean level used for the formulation is different between the two groups.

# Kruskal-Wallis test is a non-parametric test and an alternative to One-Way Anova. By parametric we mean, the data is not assumed to become from a particular distribution. The main objective of this test is used to determine whether there is a statistical difference between the medians of at least three independent groups. 
# 
# The null hypothesis (H0): The median is the same for all the data groups.
# The alternative hypothesis: (Ha): The median is not equal for all the data groups.

# In[16]:


stat, p = stats.kruskal(df['a'],df['b'],df['c'],df['d'],
                       df['e'],df['f'],df['g'],df['h'],
                       df['i'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')


# The test statistic comes out to be equal to 1707.638 and the corresponding p-value is 0.000. As the p-value is less than 0.05, we reject the null hypothesis that the median amount of additive use is the not same for all groups. Hence, We have sufficient proof to claim that the different types of additives used lead to statistically significant differences in the formulation.

# In[17]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)


# Here, standard scaler is used to re-scale the levels of all chemicals. Frankly speaking, the scaling has no effect on the sum of the squared Euclidean distances. However, in order to reduce computational cost, for simplicity, it is preferred to re-scale the dataset.

# In[18]:


pd.DataFrame(data_scaled).describe()


# In[19]:


kmeans = KMeans(n_clusters=2, init='k-means++')
kmeans.fit(data_scaled)


# In this step, the cluster is initially defined as 2 clusters. The reason is to initialize the k-means calculation, for determinig the starting point of sum of squared Euclidean distance or inertia. Init use is k-means++ to speed up the calculation convergence.

# In[20]:


kmeans.inertia_


# For 2 clusters, the inertia calculated is 1526.2793711914974.

# In[21]:


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(2,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)


# In order to pinpoint an appropriate number of clusters in this modelling, elbow method is used. Here, cluster numbers from 2 to 20 were tested, using KMeans function. Initially, empty list of SSE was defined, and later this list appended as the number of clusters tested. For each cluster number tested, the modelling fitted with scaled dataset.

# In[22]:


frame = pd.DataFrame({'Cluster':range(2,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# The result of kmeans calculation using cluster number 2 to 20, being saved as a dataframe in object called frame. It is boserved that, the decrement of inertia becoming steady from 7 onwards. Thus, by using elbow method rule of thumb, the onset of the steady decrement is selected for the optimum cluster number.

# In[23]:


get_ipython().system('pip install --upgrade kneed')


# In[24]:


from kneed import KneeLocator
elbow = KneeLocator(range(2, 20), SSE, curve="convex", direction="decreasing")


# In[25]:


elbow.elbow


# In order to firm up the selection, KneeLocator function from kneed is used. Here, same range of cluster number were tested, where the number of optimum cluster is calculated as 7.

# In[26]:


# k means using 8 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 7, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)


# In[27]:


frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()


# Based on the elbow method test, 7 clusters specified in the kmeans modelling. Using the trained model, the quantity of datapoints in each cluster calculated. It is discovered that 95 out of 214 formulations belongs in same cluster; 0. The least is cluter 5 where only 2 formulations. Thus, it can be predicted that 44% of the formulations will yield the same result.

# In[ ]:




