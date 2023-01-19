#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing all the important libraries.

# it is python library which is used to work with datasets.
import pandas as PD 
# it is python library which is used to work with arrays.
import numpy as np 
# K-means is a way to group data points without being told what to do. The algorithm divides the data points in to the K clusters by reducing the amount of difference between each cluster.
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plotmat

# importing warnings.
import warnings 
warnings.filterwarnings('ignore')    

# import scipy.
import scipy
# it is python library which is used to work with arrays.

# importing curve fit from scipy.
# from scipy.optimize import curvefit_df
# Matplotlib is a Python library that lets you make rigid, animated, and interactive visualisations. Matplotlib makes things that are easy and things that are hard possible.
import matplotlib.pyplot as plotmat
from scipy import stats 

from sklearn import metrics
#importing metrics from sklearn 
# importing label encoder from scikit learn. 
from sklearn.preprocessing import LabelEncoder

# using the elbow method to find out the clusters.
from scipy.spatial.distance import cdist 


# # K Means Clustering

# In[5]:


# Creates the function for analyse the dataset.
def read_dataset(new_file):
    forest_df = PD.read_csv(new_file, skiprows=4) # using pandas read data and skip starting 4 rows from data.
    forest_df1 = forest_df.drop(['Unnamed: 66', 'Indicator Code',  'Country Code'],axis=1) # dropping the columns.
    forest_df2 = forest_df1.set_index("Country Name")  
    forest_df2=forest_df2.T 
    forest_df2.reset_index(inplace=True) 
    forest_df2.rename(columns = {'index':'Year'}, inplace = True) 
    return forest_df1, forest_df2 

# define the path of electricity data.
forestcsv = '/content/drive/MyDrive/API_AG.LND.FRST.ZS_DS2_en_csv_v2_4770431.csv'  
full_forest_df, Transpose_data = read_dataset(forestcsv)   
full_forest_df.head() # showing starting rows. 


# In[7]:


Transpose_data.head()


# In[8]:


# Extracting 20 years of data with the help of function.
def full_forest_df2(full_forest_df): 
    full_forest_df1 = full_forest_df[['Country Name', 'Indicator Name','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010',
                                    '2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']] 
    full_forest_df2 = full_forest_df1.dropna() # drop null values from data.
    return full_forest_df2

# calling the function to extract the data. 
full_forest_df3 = full_forest_df2(full_forest_df) 
full_forest_df3.head(10) # shows starting rows from data.


# In[9]:


full_forest_df3


# In[10]:


# check null values from data.
full_forest_df3.isnull().sum()


# In[11]:



lblencoder = LabelEncoder()# define classifier for encoder.
full_forest_df3['Country Name'] = lblencoder.fit_transform(full_forest_df3['Country Name']) 
full_forest_df3.head(10) # showing 5 rows from data.


# In[12]:


X = full_forest_df3.drop(['Country Name','Indicator Name'], axis=1)
y = full_forest_df3['Country Name']  

# importing minmax scaler for normalize the data.
from sklearn.preprocessing import MinMaxScaler
scaleminmax = MinMaxScaler()# define classifier.
scaled_df_minmax = scaleminmax.fit_transform(X)# fit classifier with data.  


# # Elbow Method 

# In[13]:



Cluster = range(10) 
Meandist = list()

for k in Cluster:
    model = KMeans(n_clusters=k+1) 
    model.fit(scaled_df_minmax) 
    Meandist.append(sum(np.min(cdist(scaled_df_minmax, model.cluster_centers_, 'euclidean'), axis=1)) / scaled_df_minmax.shape[0]) 

# setting all the parameter and ploting the graph.

# define font size.
plotmat.rcParams.update({'font.size': 20})
# define figure size.
plotmat.figure(figsize=(10,7))
# set parameter for graph.
plotmat.plot(Cluster, Meandist, marker="o") 
# define xlabel.
plotmat.xlabel('Numbers of Clusters')
# define ylabel.
plotmat.ylabel('Average distance') 
# define title for graph.
plotmat.title('Choosing k with the Elbow Method'); 


# In[14]:


# define classifier for clustering.
k_means_model = KMeans(n_clusters=3, max_iter=100, n_init=10,random_state=10)
# fit classifier with data.  
k_means_model.fit(scaled_df_minmax) 
# predict model to getting the label.
predictions = k_means_model.predict(scaled_df_minmax)  


# In[15]:


predictions 


# In[16]:


# Getting the Centroids and label.
centroids = k_means_model.cluster_centers_
u_labels = np.unique(predictions) 
centroids



# In[17]:


plotmat.scatter(scaled_df_minmax[predictions==0, 0], scaled_df_minmax[predictions==0, 1], s=100, c='red', label ='First cluster')
#Plotting first cluster with the red color  
plotmat.scatter(scaled_df_minmax[predictions==1, 0], scaled_df_minmax[predictions==1, 1], s=100, c='blue', label ='Second cluster')
#Plotting Second cluster with the blue color  
plotmat.scatter(scaled_df_minmax[predictions==2, 0], scaled_df_minmax[predictions==2, 1], s=100, c='green', label ='Third cluster')
#Plotting Third cluster with the green color  
# plt.scatter(k_means_model.cluster_centers_[:, 0], k_means_model.cluster_centers_[:, 1], marker='x', s=300,c='black', label = 'Centroids')
#plotting title
plotmat.title('Clusters of Product')
#x axis label
plotmat.xlabel('X')
#y axis label
plotmat.ylabel('Y')
plotmat.show();


# In[18]:



#importing silhoette from the sklearn to evaluate the k means cluster 
sil_score = metrics.silhouette_score(scaled_df_minmax, predictions)
#evaluating the score in which passing prediction and true db_labels
print(f'Cluster Silhouette Score: {sil_score}')
#priniting the score in which passing prediction and true db_labels


# In[19]:


from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Create a Dataframe from the scaled_df_minmax data
data = {'x': scaled_df_minmax[:, 0], 'y': scaled_df_minmax[:, 1], 'cluster': predictions}
df = DataFrame(data, columns=['x', 'y', 'cluster'])

plt.figure(figsize=(10, 7))
plt.title('Parallel Coordinates Plot for 3 Clusters')
parallel_coordinates(df, 'cluster')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[20]:


# plotting the results.
plotmat.figure(figsize=(10,7))
for i in u_labels:
    plotmat.scatter(scaled_df_minmax[predictions == i , 0] , scaled_df_minmax[predictions == i , 1] , label = i)  

# define parameter for graph like color, data etc.
plotmat.scatter(centroids[:,0] , centroids[:,1] , s = 100, color = 'm') 
# define xlabel.
plotmat.xlabel('X')
# define ylabel.
plotmat.ylabel('Y')
# define title for graphs.
plotmat.title('Scatter plot for 3 Clusters with Centroids') 
# define legend for graph.
plotmat.legend()  
plotmat.show()  


# In[21]:


# creating the lists to extract all the cluster.
first_cluster=[]
second_cluster=[] 
third_cluster=[] 

# with the help of loop find out the data availabel in each cluster.
for i in range(len(predictions)):
    if predictions[i]==0:
        first_cluster.append(full_forest_df.loc[i]['Country Name']) 
    elif predictions[i]==1:
        second_cluster.append(full_forest_df.loc[i]['Country Name'])
    else:
        third_cluster.append(full_forest_df.loc[i]['Country Name'])   


# In[22]:


# showing the data present in first cluster.
First_cluster = np.array(first_cluster)
print(First_cluster)


# In[23]:


# showing the data present in second cluster.
Second_cluster = np.array(second_cluster)
print(Second_cluster)  


# In[24]:


# showing the data present in third cluster.
Third_cluster = np.array(third_cluster)
print(Third_cluster)  


# In[25]:


first_cluster = First_cluster[50] 
print('Country name :', first_cluster)
Maldives = full_forest_df3[full_forest_df3['Country Name']==8]  
Maldives = np.array(Maldives)  
Maldives = np.delete(Maldives,1) 
Maldives    


# In[26]:


second_cluster = Second_cluster[24] 
print('Country name :', second_cluster) 
Guyana = full_forest_df3[full_forest_df3['Country Name']==0] 
Guyana = np.array(Guyana)  
Guyana = np.delete(Guyana,1) 
Guyana  


# In[27]:


third_cluster = Third_cluster[53] 
print('Country name :', third_cluster) 
Lao_PDR = full_forest_df3[full_forest_df3['Country Name']==1] 
Lao_PDR= np.array(Lao_PDR)  
Lao_PDR = np.delete(Lao_PDR,1) 
Lao_PDR  


# In[28]:


year=list(range(2000,2022))

plt.figure(figsize=(22,8))
plt.rcParams.update({'font.size': 20})

plt.subplot(131)
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Maldives')
plt.bar(year, Maldives, color='g')

plt.subplot(132)
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Guyana')
plt.bar(year, Guyana, color='b')

plt.subplot(133)
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Lao PDR')
plt.bar(year, Lao_PDR, color='r')


# # Curve Fitting

# In[29]:


# calling the function to extract the data. 
full_forest_df4 = full_forest_df2(full_forest_df) 
full_forest_df4.head(10) 


# In[30]:


full_forest_df4['Country Name'].unique()


# In[31]:


# check shape of data.
full_forest_df4.shape 


# In[32]:


# check null values in data.
full_forest_df4.isnull().sum()


# In[33]:


# selecting all columns and convert into array.
x = np.array(full_forest_df3.columns) 
# dropping some columns.
x = np.delete(x,0) 
x = np.delete(x,0) 
# convert data type as int.
x = x.astype(np.int)

# selecting all the data for urban population and india.
curvefit_df = full_forest_df4[(full_forest_df4['Indicator Name']=='Forest area (% of land area)') & (full_forest_df4['Country Name']=='Africa Western and Central')]   

# convert into array.
y = np.array(curvefit_df)
# dropping some columns.
y = np.delete(y,0) 
y = np.delete(y,0)
# convert data type as int.
#y = y.astype(np.int) 


# In[34]:


curvefit_df


# In[35]:


x


# In[36]:


y


# In[37]:


from scipy.optimize import curve_fit

# Define the function to be fitted (linear function y = mx + c)
def linear_func(x, m, c):
    return m*x + c

def create_curve_fit(x,y): 

    # Perform curve fitting
    popt, pcov = curve_fit(linear_func, x, y) 
    labels=x
    # Extract the fitted parameters and their standard errors
    m, c = popt
    m_err, c_err = np.sqrt(np.diag(pcov)) 

    # Calculate the lower and upper limits of the confidence range
    conf_int = 0.95  # set the confidence interval as 95%
    alpha = 1.0 - conf_int 
    m_low, m_high = scipy.stats.t.interval(alpha, len(x)-2, loc=m, scale=m_err)
    c_low, c_high = scipy.stats.t.interval(alpha, len(x)-2, loc=c, scale=c_err)

#     # Plot the best-fitting function and the confidence range.
    plotmat.figure(figsize=(12,6)) #define figure size.
    plotmat.rcParams.update({'font.size': 10}) #define fontsize.
    plotmat.plot(x, y, 'bo', label='Data') #set data for graph.
    plotmat.plot(x, linear_func(x, m, c), 'b', label='Fitted function')
    plotmat.fill_between(x, linear_func(x, m_low, c_low), linear_func(x, m_high, c_high), color='gray', alpha=0.1, label='Confidence Interval') # set all the parameter.
    plotmat.title('Curve Fitting') # define title for graph.
    plotmat.xlabel('Years')
    plotmat.xticks(x, labels, rotation ='vertical')
# define xlabel.
    plotmat.ylabel('Land Area') # define ylabel. 
    plotmat.legend() # set legend in graph.
    plotmat.show() 
    



# In[38]:


create_curve_fit(x,y)


# In[38]:





# In[38]:












# In[38]:




