rm(list=ls())
setwd("C:/Users/BATMAN/Desktop/edWisor Bike Renting")
getwd()

#install.packages("Metrics")
#install.packages("usdm")
library(corrplot)
library(rpart)
library(MASS)
library(rsq)
library(caret)
library(Metrics)
library(usdm)
library(DMwR)
library(grid)
library(e1071)
library(class)
library(randomForest)
library(ggplot2)


#Loading data from CSV
#*********************
df =  read.csv("C:/Users/BATMAN/Desktop/edWisor Bike Renting/day.csv") 

#Missing value analysis
#*****************
#Check missing vales 
sum(is.na(df))

#Outlier Analysis
#*****************
#check for outliers
numeric_index = sapply(df,is.numeric) #selecting only numeric
numeric_data = df[,numeric_index] 
numeric_predictors = subset(numeric_data, select = c('temp','atemp','hum','windspeed','registered'))
cnames = colnames(numeric_predictors)
#loop to plot box-plot for each variable
for(i in cnames)
{
  boxplot(df[i],c)
}
#replacing outliers with Null value
for(i in cnames)
{
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% val] = NA
}
#imputing nill value with Knn-imputation
df = knnImputation(df, k=3)

df = subset(df, select = -c(instant,dteday,casual,registered))
df = subset(df, select = -c(dteday))

#feature selection
imppred <- randomForest(cnt ~ ., data = df,ntree = 100, keep.forest = FALSE, importance = TRUE)
importance(imppred, type = 1)
varImpPlot(imppred)
#correlation matrix
symnum(cor(df))




#-------------------------------
#Exploratory Data Analysis (EDA)
#-------------------------------

#Distribution of contineous variables
hist(df$cnt,col='gold')
hist(df$temp,col='gold')
hist(df$atemp,col='gold')
hist(df$hum,col='gold')
hist(df$windspeed,col='gold')

#Boxplots with notch
----------------------
#year
boxplot(df$cnt ~ df$yr, data = df, xlab = "Year",ylab = "Bike rent count", 
          main = "Bike rental Data",notch = TRUE, varwidth = TRUE, col = c("red","green","blue"))
#Here year - > (0:2011, 1:2012)
#observations : bike renting count was more in 2012 as compared to 2011
----------------------
#season
boxplot(df$cnt ~ df$season, data = df,  xlab = "Season",ylab = "Bike rent count", 
          main = "Bike rental Data",notch = TRUE, varwidth = TRUE, col = c("red","green","blue"))
#Here Season - > (1:springer, 2:summer, 3:fall, 4:winter)
#observations : bike renting count was maximum in fall and summer & least in springer
#---------------------
#month
boxplot(df$cnt ~ df$mnth, data = df, xlab = "Month",ylab = "Bike rent count", 
        main = "Bike rental Data",notch = TRUE, varwidth = TRUE, col = c("red","green","blue"))
#Here : Month (1 to 12) are (jan-Dec)
#observations : bike renting count was more in summer and fall months as compared to winter months
#---------------------
#weekday
boxplot(df$cnt ~ df$weekday, data = df,  xlab = "Weekday",ylab = "Bike rent count", 
        main = "Bike rental Data",notch = TRUE, varwidth = TRUE, col = c("red","green","blue"))
#Here ->  Month (0:6) are (Sun:Sat))
#observations : there is no strong relation between weekday and bike count
#---------------------


#Categorical data vs target variable "cnt"
#bar graphs
#---------------
#weekday plot
counts <- table(df$yr, df$weekday)
barplot(counts, main="Bike rent count Distribution by weekday and year",xlab="Weekday", col=c("darkblue","red"),legend = rownames(counts), beside=TRUE)
#workingday plot
counts <- table(df$yr, df$workingday)
barplot(counts, main="Bike rent count Distribution by workingday and year",xlab="Workingday", col=c("darkblue","red"),legend = rownames(counts), beside=TRUE)
#holiday plot
counts <- table(df$yr, df$holiday)
barplot(counts, main="Bike rent count Distribution by holidy and year",xlab="Holiday", col=c("darkblue","red"),legend = rownames(counts), beside=TRUE)
#---------------

#Scatterp-plot
#understanding liner relationship between contineous variables
#temp
plot(df$temp,df$cnt,xlab = "temp",ylab = "Count",main = "Weekday vs count")
#atemp
plot(df$atemp,df$cnt,xlab = "atemp",ylab = "Count",main = "Weekday vs count")
#humidity
plot(df$hum,df$cnt,xlab = "Humidity",ylab = "Count",main = "Weekday vs count")
#windspeed
plot(df$windspeed,df$cnt,xlab = "windspeed",ylab = "Count",main = "Weekday vs count")
#registered
plot(df$registered,df$cnt,xlab = "registered",ylab = "Count",main = "Weekday vs count")
#casual
plot(df$casual,df$cnt,xlab = "casual",ylab = "Count",main = "Weekday vs count")



#feature selection
#*****************
#correlation analysis using corrplot
w = subset(numeric_data, select = c('temp','atemp','hum','windspeed'))
x = cor(w)
corrplot(x, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)
df = subset(df, select = -c(instant,dteday,casual,registered))

#feature importance check
#Create an importance based on mean decreasing gini
fit_rf = randomForest(cnt~., data=df)
importance(fit_rf)

#
df = subset(df, select = -c(dteday))
df = subset(df, select = -c(casual))
df = subset(df, select = -c(holiday))
df = subset(df, select = -c(mnth))

df = subset(df, select = -c(windspeed))
df = subset(df, select = -c(season))
df = subset(df, select = -c(weekday))
df = subset(df, select = -c(weathersit))


lrmodel <- lm(cnt ~ ., data = df)
summary(lrmodel)



#Sampling
#Splitting into train and test
n = nrow(df)
trainIndex = sample(1:n, size = round(0.8*n), replace=FALSE)
train = df[trainIndex ,]
test = df[-trainIndex ,]

#R-Square value
rsq <- function (x, y) cor(x, y) ^ 2

#Randomforest
library(randomForest)
set.seed(71) 
rf <-randomForest(cnt ~.,data=train, ntree=500) 
test_predictions = predict(rf,test[,-12])
train_predictions = predict(rf,train[,-12])
a = mape(test[,12],test_predictions)
b = mape(train[,12],train_predictions)
c = rmse(test[,12],test_predictions)
d = rmse(train[,12],train_predictions)
e = rsq(test[,12],test_predictions)
f = rsq(train[,12],train_predictions)
cat("Train: MAPE, rmse, rsq =  ", b,d,f, "\nTest:  MAPE, rmse, rsq =  ", a,c,e)
#Linear rigression
lm_model = lm(cnt ~.,data=train)
test_predictions = predict(lm_model,test[,-12])
train_predictions = predict(lm_model,train[,-12])
a = mape(test[,12],test_predictions)
b = mape(train[,12],train_predictions)
c = rmse(test[,12],test_predictions)
d = rmse(train[,12],train_predictions)
e = rsq(test[,12],test_predictions)
f = rsq(train[,12],train_predictions)
cat("Train: MAPE, rmse, rsq =  ", b,d,f, "\nTest:  MAPE, rmse, rsq =  ", a,c,e)
#D-Tree
fit = rpart(cnt ~.,data=train,method = 'anova')
test_predictions = predict(fit,test[,-12])
train_predictions = predict(fit,train[,-12])
a = mape(test[,12],test_predictions)
b = mape(train[,12],train_predictions)
c = rmse(test[,12],test_predictions)
d = rmse(train[,12],train_predictions)
e = rsq(test[,12],test_predictions)
f = rsq(train[,12],train_predictions)
cat("Train: MAPE, rmse, rsq =  ", b,d,f, "\nTest:  MAPE, rmse, rsq =  ", a,c,e)

#*****************************************
