# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 07:21:49 2020

@author: ameyk
"""


# **************** Import the libraries ************************
import pandas as pd
import numpy as np
#from factor_analyzer import FactorAnalyzer
from sklearn import preprocessing
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# **************** Read database **********************
datapath = 'credit-card-data.csv'
data = pd.read_csv(datapath)
print(data.head())
#print(data.columns)
#print(data.describe())

#****************** Dataset Cleaning ****************************
# checking for the null values
data.isnull().sum()
# Minimum payments and Credit Limit have null values
print(data["MINIMUM_PAYMENTS"].describe())
print(data["CREDIT_LIMIT"].describe())
#fill the na with median values
data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].median(),inplace=True)
data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median(),inplace=True)
#checking all null values are treated
print(data.isnull().sum())

# **************** Deriving KPIs **********************
# 1. Monthly average purchase
data["MONTHLY_AVG_PURCHASE"] = data["PURCHASES"] / 12
print(data["MONTHLY_AVG_PURCHASE"].head(5))

# 2. Average Cash Advance Amount
data["MONTHLY_AVG_CASH_ADVAANCE"] = data["CASH_ADVANCE"] / 12
print(data.head(5))

# 3. Puchase Type of Customer
print(data.loc[:,['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']].head(20))
# we can see there are four types of customers
#1. Both type of purchase amount is zero (None Category)
#2. Puchases type is completely ONEOFF Purchases (Installment Purchase = 0)
#3. Puchases type is completely Instalment Purchases (ONEOFF Purchase = 0)
#4. Purchases are done using both the types (Instalment purchase >0  and ONEOFF Purchase > 0)
# dividing to 4 categories - ONEOFF, INSTALMENT, BOTH, NONE

def purchaseCategory(data):   
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'NONE'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']>0):
         return 'BOTH'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'ONEOFF'
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']>0):
        return 'INSTALMENT'

data['PURCHASE_TYPE']=data.apply(purchaseCategory,axis=1)
data['PURCHASE_TYPE'].value_counts()
#BOTH          2774
#INSTALMENT    2260
#NONE          2042
#ONEOFF        1874
# 4. Balance to Credit Limit Ratio
data["BALANCE_CREDIT_LIMIT"] = data["BALANCE"] / data["CREDIT_LIMIT"]
print(data.head(5))


# 5. Payments to minimum payments ratio
data["PAYMENTS_MINIMUM_PAYMENTS"] = data["PAYMENTS"]/data["MINIMUM_PAYMENTS"]
print(data.head(5))

# write data to csv which can be used in Tableau
data.to_csv("credit_card_data_with_kpi.csv")

# check for ranges of diffrent columns
data.describe()
# we can see there are various ranges of columns in data 
#Perform feature scaling. 

#removing categorical type and Cust ID column
data_scaled = data
data_scaled = data_scaled.drop(['CUST_ID','PURCHASE_TYPE'],axis=1)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data_scaled.values)
data_scaled = pd.DataFrame(x_scaled,columns=data_scaled.columns)
print(data_scaled.columns)

#converting categorical variables to dummies
data_scaled=pd.concat([data_scaled,pd.get_dummies(data['PURCHASE_TYPE'])],axis=1)
data_scaled.head()

# **************************** Factor Analysis Using PCA ************************
# correlation heatmap
sns.heatmap(data_scaled.corr())
var_ratio={}
N = range(1,20)
for n in N:
    pc=PCA(n_components=n)
    cr_pca=pc.fit(data_scaled)
    var_ratio[n]=sum(cr_pca.explained_variance_ratio_)
    
print(var_ratio)
plt.plot(pd.Series(var_ratio), 'bx-')
plt.xlabel('N Components')
plt.ylabel('Variance Ratio')
plt.title('Variance explained by number of components')
plt.show()
#pd.Series(var_ratio).plot()
#{1: 0.37438252783520687, 2: 0.5950617518758892, 3: 0.7607727267071046, 4: 0.8197626927335029, 5: 0.8700259883682393, 6: 0.9057770436314251, 7: 0.9329753852335672, 8: 0.9580878232256673, 9: 0.9746942559046184, 10: 0.9833847721705338, 11: 0.988652497515903, 12: 0.9918352140492432, 13: 0.9946891219988491, 14: 0.9964136901495445, 15: 0.9974351773055743, 16: 0.9980485729537011, 17: 0.9986304560751879, 18: 0.9991417074359745, 19: 0.9996152853267428}
#based on the values, we can conclude that 5 variables explains the 88% data variance. So we select the 5 components
pca_analysis=PCA(n_components=5).fit(data_scaled)
reduced_data=pca_analysis.fit_transform(data_scaled)
reduced_scaled_data = pd.DataFrame(reduced_data)
print(reduced_scaled_data.head())
col_list = data_scaled.columns
print(col_list)

variance_details = pd.DataFrame(pca_analysis.components_.T, columns=['PC_' +str(i) for i in range(5)],index=col_list)
print(variance_details)

# =============================================================================
#                                       PC_0      PC_1      PC_2      PC_3      PC_4
# BALANCE                          -0.011619 -0.042876 -0.035515 -0.131301 -0.117336
# BALANCE_FREQUENCY                 0.062510 -0.085298 -0.069840 -0.234271 -0.610312
# PURCHASES                         0.025404 -0.021201  0.002920  0.018269 -0.026424
# ONEOFF_PURCHASES                  0.015522 -0.024410  0.008450  0.019372 -0.023693
# INSTALLMENTS_PURCHASES            0.027257 -0.001960 -0.008941  0.004717 -0.014664
# CASH_ADVANCE                     -0.013756 -0.008132 -0.015621 -0.017086 -0.024739
# PURCHASES_FREQUENCY               0.518578  0.040013  0.097841  0.016997 -0.353856
# ONEOFF_PURCHASES_FREQUENCY        0.185572 -0.330033  0.169852  0.191974 -0.325359
# PURCHASES_INSTALLMENTS_FREQUENCY  0.487224  0.187284 -0.134460 -0.107378 -0.164502
# CASH_ADVANCE_FREQUENCY           -0.060053 -0.037454 -0.064216 -0.082892 -0.106047
# CASH_ADVANCE_TRX                 -0.015989 -0.012335 -0.019366 -0.027398 -0.032273
# PURCHASES_TRX                     0.056845 -0.029375 -0.004115  0.010904 -0.056960
# CREDIT_LIMIT                      0.024514 -0.057156 -0.002550  0.013837 -0.062480
# PAYMENTS                          0.010127 -0.020636 -0.006764  0.020619 -0.029431
# MINIMUM_PAYMENTS                 -0.000027 -0.000285 -0.003430 -0.018916 -0.013759
# PRC_FULL_PAYMENT                  0.129212  0.084969  0.055587  0.901787 -0.036925
# TENURE                            0.033167 -0.030084  0.000655 -0.103889 -0.060625
# MONTHLY_AVG_PURCHASE              0.025404 -0.021201  0.002920  0.018269 -0.026424
# MONTHLY_AVG_CASH_ADVAANCE        -0.013756 -0.008132 -0.015621 -0.017086 -0.024739
# BALANCE_CREDIT_LIMIT             -0.007467 -0.005678 -0.008784 -0.034907 -0.025609
# PAYMENTS_MINIMUM_PAYMENTS         0.000169  0.000802 -0.000177  0.002107  0.002332
# BOTH                              0.440742 -0.508083 -0.263791 -0.055025  0.416149
# INSTALMENT                        0.163992  0.730647  0.071448 -0.075732  0.065084
# NONE                             -0.409396 -0.031825 -0.550268  0.180037 -0.368722
# ONEOFF                           -0.195338 -0.190739  0.742610 -0.049279 -0.112511
# 
# =============================================================================

# *********************************************** Customer Segmentation *******************************************
#popularly used techniques for segmentation/ clsutering are 
# 1. K-means clustering
# 2. Hierarchical Clustering
# K-means is most basic and simple techinique for clustering. We will use that
#here we will use sum of squared distances to find out number of clusetrs (most baseic method)
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(reduced_scaled_data)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# we can see the error is decreasing steeply till k value 4 and after k = 6 it is not decreasing significantly
#selecting k = 4
km_4=KMeans(n_clusters=4,random_state=123)
km_4.fit(reduced_scaled_data)
print(km_4.labels_)

pd.Series(km_4.labels_).value_counts()
# =============================================================================
# 2    2774
# 0    2260
# 1    2042
# 3    1874
# =============================================================================


# Conactenating labels found through Kmeans with data 
print(data.columns)
cluster_df_4=pd.concat([data,pd.Series(km_4.labels_,name='Cluster_4')],axis=1)
cluster_df_4.head()
print(cluster_df_4.columns)

# write to csv for visualisation in Tableau
cluster_df_4.to_csv("clustered_credit_card.csv")

# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
cluster_4=cluster_df_4.groupby('Cluster_4').apply(lambda x: x[x.columns].mean()).T
print(cluster_4)

# =============================================================================
# Cluster_4                                   0            1            2            3
# BALANCE                            845.278262  2147.759969  1805.968182  1438.761601
# PURCHASES                          536.904217     0.000000  2267.805807   786.759029
# ONEOFF_PURCHASES                     0.000000     0.000000  1379.884427   786.827679
# INSTALLMENTS_PURCHASES             537.878469     0.000000   888.049776     0.000000
# CASH_ADVANCE                       419.959034  1988.467370   764.226676   870.530348
# CREDIT_LIMIT                      3371.864329  4025.987594  5738.829463  4515.920572
# MONTHLY_AVG_PURCHASE                44.742018     0.000000   188.983817    65.563252
# MONTHLY_AVG_CASH_ADVAANCE           34.996586   165.705614    63.685556    72.544196
# BALANCE_CREDIT_LIMIT                 0.271678     0.573771     0.353548     0.381074
# PAYMENTS_MINIMUM_PAYMENTS           13.259004    10.087451     7.236982     5.571082
# Cluster_4                            0.000000     1.000000     2.000000     3.000000
# =============================================================================

# we can see from the avrage value
# 1. Cluster 0 - Instalment Purchases are higher, balance to credit limit ratios is low, Credit limit is low, cash advance is low, monthly avergae purchase is low, payments to minimum payments ratio is high
#From the above observations we can say user of cluster 0 is having low credit score and high risk.
# 2. Cluster 1 - Instalment Purchases and ONEOFF purchases both are zero, balance to credit limit ratio high, Credit limit is average, cash advance is high, monthly avergae purchase is very low, payments to minimum payments ratio is average
#From the above observations we can say user of cluster 1 is using credit card only for cash purposes and paying the amount in time.
# 3. Cluster 2 - Instalment Purchases and ONEOFF purchases both are average, balance to credit limit ratio average, Credit limit is high, cash advance is low, monthly avergae purchase is high, payments to minimum payments ratio is average
#From the above observations we can say user of cluster 2 is using credit card regularly for purchasing, paying the amount in time and having a good credit score.
# 4. Cluster 3 - ONEOFF purchases are higher, balance to credit limit ratio average, Credit limit is average, cash advance is low, monthly avergae purchase is average, payments to minimum payments ratio is very low
## 3. Cluster 2 - Instalment Purchases and ONEOFF purchases both are average, balance to credit limit ratio average, Credit limit is high, cash advance is low, monthly avergae purchase is high, payments to minimum payments ratio is average
#From the above observations we can say user of cluster 2 is using credit card regularly for purchasing, paying the amount in time and having a good credit score.



km_5=KMeans(n_clusters=5,random_state=123)
km_5.fit(reduced_scaled_data)
print(km_5.labels_)

pd.Series(km_5.labels_).value_counts()
# =============================================================================
# 2    2260
# 1    2042
# 3    1874
# 4    1724
# 0    1050
# =============================================================================


# Conactenating labels found through Kmeans with data 
print(data.columns)
cluster_df_5=pd.concat([data,pd.Series(km_5.labels_,name='Cluster_5')],axis=1)
cluster_df_5.head()
print(cluster_df_5.columns)

# write to csv for visualisation in Tableau
cluster_df_5.to_csv("Five_clustered_credit_card.csv")

# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
cluster_5=cluster_df_5.groupby('Cluster_5').apply(lambda x: x[x.columns].mean()).T
print(cluster_5)

# =============================================================================
# Cluster_5                                   0            1            2            3            4
# ONEOFF_PURCHASES                   662.761648     0.000000     0.000000   786.827679  1816.647140
# INSTALLMENTS_PURCHASES             362.246533     0.000000   537.878469     0.000000  1208.289571
# CASH_ADVANCE                       848.914901  1988.467370   419.959034   870.530348   712.647420
# CREDIT_LIMIT                      4703.345647  4025.987594  3371.864329  4515.920572  6369.489559
# MONTHLY_AVG_PURCHASE                85.390844     0.000000    44.742018    65.563252   252.076985
# MONTHLY_AVG_CASH_ADVAANCE           70.742908   165.705614    34.996586    72.544196    59.387285
# BALANCE_CREDIT_LIMIT                 0.379697     0.573771     0.271678     0.381074     0.337623
# PAYMENTS_MINIMUM_PAYMENTS            5.337080    10.087451    13.259004     5.571082     8.394115
# =============================================================================

# we can see in cluster 0 and cluster 3 major difference is only the instalment purchase amount
# Also in cluster 0 and cluster 4 both are having both purchase type users, mainly the difference is mponthly avergae puchase amount 
# so in this scenario clusters are overlapping and will not be able to give properly distinguish

km_6=KMeans(n_clusters=6,random_state=123)
km_6.fit(reduced_scaled_data)
print(km_6.labels_)

print(pd.Series(km_6.labels_).value_counts())
# =============================================================================
# 2    2042
# 0    1874
# 1    1724
# 3    1338
# 5    1050
# 4     922
# =============================================================================

# Conactenating labels found through Kmeans with data 
print(data.columns)
cluster_df_6=pd.concat([data,pd.Series(km_6.labels_,name='Cluster_6')],axis=1)
cluster_df_6.head()
print(cluster_df_6.columns)

# write to csv for visualisation in Tableau
cluster_df_6.to_csv("Six_clustered_credit_card.csv")

# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
cluster_6 = cluster_df_6.groupby('Cluster_6').apply(lambda x: x[x.columns].mean()).T
print(cluster_6)

# =============================================================================
# Cluster_6                                   0            1            2            3            4            5
# ONEOFF_PURCHASES                   786.827679  1816.647140     0.000000     0.000000     0.000000   662.761648
# INSTALLMENTS_PURCHASES               0.000000  1208.289571     0.000000   694.489028   310.606312   362.246533
# CREDIT_LIMIT                      4515.920572  6369.489559  4025.987594  3148.712461  3695.700771  4703.345647
# MONTHLY_AVG_PURCHASE                65.563252   252.076985     0.000000    57.780597    25.820523    85.390844
# MONTHLY_AVG_CASH_ADVAANCE           72.544196    59.387285   165.705614    30.816006    41.063415    70.742908
# BALANCE_CREDIT_LIMIT                 0.381074     0.337623     0.573771     0.298183     0.233214     0.379697
# PAYMENTS_MINIMUM_PAYMENTS            5.571082     8.394115    10.087451    11.408281    15.944759     5.337080
# =============================================================================

# when we analyze for six clusetrs also there aree overlapping clusters like cluster 3 and 4. Where there is no major difference in attributes.
# so selecting the clusters as 4.


