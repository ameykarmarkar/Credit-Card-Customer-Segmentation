
# *************************** load necessary libraries ************************
library(Hmisc)
library(dplyr)
library(fastDummies)
library(factoextra)
# *************************** read database ******************************
datapath <- 'credit-card-data.csv'
data <- read.csv(datapath)

# *************************** Dataset Cleaning **************************
# check for the null values
lapply(data,function(x) { length(which(is.na(x)))})
#minimum payments and credit limit have null values
print(describe(data$MINIMUM_PAYMENTS))
print(describe(data$CREDIT_LIMIT))

#fill na values with median
data$MINIMUM_PAYMENTS[which(is.na(data$MINIMUM_PAYMENTS))] <- median(data$MINIMUM_PAYMENTS, na.rm = TRUE)
data$CREDIT_LIMIT[which(is.na(data$CREDIT_LIMIT))] <- median(data$CREDIT_LIMIT, na.rm = TRUE)
lapply(data,function(x) { length(which(is.na(x)))})

# **************************** Deriving KPI's *****************************
# 1. Monthly Average Purchase
data$MONTHLY_AVG_PURCHASE = data$PURCHASES / 12

# 2. Monthly Average Cash Advance
data$MONTHLY_AVG_CASH_ADVANCE = data$CASH_ADVANCE / 12

# 3. Purchase Type of Customer
select(data, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES)
# we can see there are four types of customers
#1. Both type of purchase amount is zero (None Category)
#2. Puchases type is completely ONEOFF Purchases (Installment Purchase = 0)
#3. Puchases type is completely Instalment Purchases (ONEOFF Purchase = 0)
#4. Purchases are done using both the types (Instalment purchase >0  and ONEOFF Purchase > 0)
# dividing to 4 categories - ONEOFF, INSTALMENT, BOTH, NONE

purchaseCategory <- function(creditData) {
  if (as.integer(creditData[1]) == 0 & as.integer(creditData[2]) == 0) {
    return('NONE')
  } 
  if (as.integer(creditData[1]) > 0 & as.integer(creditData[2]) > 0) {
    return('BOTH')
  } 
  if (as.integer(creditData[1]) > 0 & as.integer(creditData[2]) == 0) {
    return('ONEOFF')
  } 
  if (as.integer(creditData[1]) == 0 & as.integer(creditData[2]) > 0) {
    return('INSTALMENT')
  } 
}

data$PURCHASE_TYPE <- unlist(apply(data[,c(5,6)], 1, purchaseCategory))

lapply(data,function(x) { length(which(is.na(x)))})
purchase_type_counts <- table(unlist(data$PURCHASE_TYPE))

# 4. Balance to credit limit ratio
data$BALANCE_CREDIT_LIMIT = data$BALANCE / data$CREDIT_LIMIT

# 5. Payments to minimum payments ratio
data$PAYMENTS_MINIMUM_PAYMENTS = data$PAYMENTS / data$MINIMUM_PAYMENTS

# write data to csv which can be used in Tableau
write.csv(data, "credit_card_data_R.csv")

# check for ranges of diffrent columns
describe(data)
# we can see there are various ranges of columns in data 
#Perform feature scaling. 

#removing categorical type and Cust ID column
data_scaled <- data
data_scaled <- subset(data_scaled, select = -c(CUST_ID,PURCHASE_TYPE) )


normalize <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

# To get a vector, use apply instead of lapply
data_scaled <- as.data.frame(apply(data_scaled,2, normalize))
print(data_scaled)

data_scaled$PURCHASE_TYPE <- data$PURCHASE_TYPE
data_scaled <- dummy_cols(data_scaled)
data_scaled <- subset(data_scaled, select = -c(PURCHASE_TYPE) )

# **************************** Factor Analysis Using PCA ************************
credit.pca <- prcomp(data_scaled, scale = FALSE)
fviz_eig(credit.pca)
# Eigenvalues
eig.val <- get_eigenvalue(credit.pca)
eig.val
# we can see that with the help of 5 dimensions cumulative variance explained is more than 85%. So selecting 5 components
# eigenvalue variance.percent cumulative.variance.percent
# Dim.1  5.288566e-01     3.744343e+01                    37.44343
# Dim.2  3.120142e-01     2.209083e+01                    59.53426
# Dim.3  2.337483e-01     1.654955e+01                    76.08380
# Dim.4  8.332110e-02     5.899194e+00                    81.98300
# Dim.5  7.089399e-02     5.019345e+00                    87.00234
# Dim.6  5.049442e-02     3.575041e+00                    90.57738
# Dim.7  3.846908e-02     2.723639e+00                    93.30102
# Dim.8  3.543037e-02     2.508496e+00                    95.80952
# Dim.9  2.345100e-02     1.660348e+00                    97.46987
# Dim.10 1.227481e-02     8.690655e-01                    98.33893
# Dim.11 7.434748e-03     5.263855e-01                    98.86532
# Dim.12 4.494295e-03     3.181993e-01                    99.18352
# Results for Variables
res.var <- get_pca_var(credit.pca)
res.var$coord          # Coordinates
#res.var$contrib        # Contributions to the PCs
#res.var$cos2           # Quality of representation 
# Results for individuals
res.ind <- get_pca_ind(credit.pca)
#res.ind$coord          # Coordinates
#res.ind$contrib        # Contributions to the PCs
#res.ind$cos2           # Quality of representation 

reduced_data <- res.ind$contrib[,0:5]
print(head(reduced_data))

reduced_data <- as.data.frame(reduced_data)


## K - Means Clustering
library(purrr)
set.seed(123)
# function to calculate total intra-cluster sum of square 
iss <- function(k) {
  kmeans(reduced_data,k,iter.max=100,nstart=100,algorithm="Lloyd" )$tot.withinss
}
k.values <- 1:10
iss_values <- map_dbl(k.values, iss)
plot(k.values, iss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total intra-clusters sum of squares")
# we can see the error is decreasing steeply till k value 4 and after k = 6 it is not decreasing significantly
#selecting k = 4

cluster_four <- kmeans(reduced_data,4)
print(head(cluster_four$cluster))
data_clustereed <-cbind(data,km_clust_4=cluster_four$cluster )
View(data_clustereed)
print(aggregate(data_clustereed, list(data_clustereed$km_clust_4), mean))


# ONEOFF_PURCHASES	INSTALLMENTS_PURCHASES	CASH_ADVANCE	CREDIT_LIMIT	MONTHLY_AVG_PURCHASE	MONTHLY_AVG_CASH_ADVANCE	BALANCE_CREDIT_LIMIT	PAYMENTS_MINIMUM_PAYMENTS	km_clust_4
# 1750.1794	745.7722	217.0129	5789.305	207.99172	18.0844	0.05048288	22.497199	1
# 120.4653	490.5163	487.3922	3609.796	50.83612	40.61601	0.29755745	12.235564	2
# 334.348	0	1476.1832	4203.572	27.85941	123.01527	0.50406182	6.121101	3
# 1330.574	948.4437	941.3285	5783.471	189.91724	78.44404	0.39771938	6.360055	4

# For cluster 1 - ONEOFF Purchase Value is high, Installments Purchase Value is average, Cash Advance is low, Monthly AVerage purchase is high, Monthly Cash Advance aamount is very low, Balance to credit limit ratio is very low, Payments to minimum payments ratio us very high
# this type of customer is using credit card at full limit and majorly using oneoff type of purchases
# For cluster 2 - Oneoff purchase is valye is low, Instalments purchase value is low, Monthly Avg Purchase Value is average, Cash Advance is low, Balance to credit limit is average Payments to minimum payments ratio is average
#users are mostly using installment payment type also they are repaying the amount regularly.
#For cluster 3 - ONEOFF Purchse value low, Instalment purchase value very low. Monthly AVrage Cash Advance Value is very high.
# users are using credit card mostly for the cash purpose only
# For cluster 4 - Users are using both type of payment ioptions also. Also Payments to minimum payments ratio is low. Customers are having good credit score.


# write data to csv which can be used in Tableau
write.csv(data_clustereed, "credit_card_four_clustered_data_R.csv")

cluster_five <- kmeans(reduced_data,5)
print(head(cluster_five$cluster))
data_clustereed <-cbind(data,km_clust_5=cluster_five$cluster )
View(data_clustereed)
print(aggregate(data_clustereed, list(data_clustereed$km_clust_5), mean))

# PURCHASES	ONEOFF_PURCHASES	INSTALLMENTS_PURCHASES	CASH_ADVANCE	PURCHASES_FREQUENCY	MONTHLY_AVG_PURCHASE	MONTHLY_AVG_CASH_ADVANCE	BALANCE_CREDIT_LIMIT	PAYMENTS_MINIMUM_PAYMENTS	km_clust_5
# 1058.2596	555.1674	503.0984	1483.2011	0.4027983	88.1883	123.60009	0.49955912	7.014424	1
# 2517.1245	1765.3897	751.7867	217.2918	0.6948889	209.76037	18.10765	0.05083347	22.685412	2
# 877.6426	367.7137	510.8269	622.6826	0.7111137	73.13689	51.89022	0.34341289	11.674294	3
# 619.7524	507.1081	112.756	295.5699	0.4159012	51.64603	24.63082	0.15268638	7.619067	4
# 671.4858	671.5726	0	1004.9525	0.247707	55.95715	83.74604	0.43761407	4.466703	5

# we can see from the mean values that some of the clusters are overlapping. So we will keep the clusters value equal to 4. 

# write data to csv which can be used in Tableau
write.csv(data_clustereed, "credit_card_five_clustered_data_R.csv")
