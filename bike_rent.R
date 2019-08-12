#importing libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','fastDummies')

#reading the csv file 
dataset <- read.csv('day.csv')
bike_rent <- dataset

#dimension of the dataset
dim(dataset)
#structure of the dataset
str(dataset)

##Eploratory data analysis
dataset$holiday=as.factor(dataset$holiday)
dataset$weekday=as.factor(dataset$weekday)
dataset$workingday=as.factor(dataset$workingday)
dataset$weathersit=as.factor(dataset$weathersit)
dataset$season=as.factor(dataset$season)
dataset$mnth=as.factor(dataset$mnth)
dataset$yr=as.factor(dataset$yr)
dataset=subset(dataset,select = -c(instant,casual,registered))
d1=unique(dataset$dteday)
df=data.frame(d1)
dataset$dteday=as.Date(df$d1,format="%Y-%m-%d")
df$d1=as.Date(df$d1,format="%Y-%m-%d")
dataset$dteday=format(as.Date(df$d1,format="%Y-%m-%d"), "%d")
dataset$dteday=as.factor(dataset$dteday)

str(dataset)

##Missing value Analysis
missing_values = data.frame(sapply(dataset,function(x){sum(is.na(x))}))


###Outlair Analysis####
#selecting only numeric
numeric_count = sapply(dataset,is.numeric) 

numeric_data = dataset[,numeric_count]

clnames = colnames(numeric_data)

for (i in 1:length(clnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (clnames[i]), x = "cnt"), data = subset(dataset))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=clnames[i],x="cnt")+
           ggtitle(paste("Box plot of count for",clnames[i])))
}


gridExtra::grid.arrange(gn1,gn2,ncol=3)
gridExtra::grid.arrange(gn3,gn4,ncol=2)


##Feature Selection
#Correlation plot

corrgram(dataset, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

## Dimension Reduction
dataset = subset(dataset,select = -c(atemp))

##Devloping The Models

#splitting the dataset into train and test data
#rmExcept("dataset")
train_count = sample(1:nrow(dataset), 0.8 * nrow(dataset))
train = dataset[train_count,]
test = dataset[-train_count,]

##Decission Tree
Dt = rpart(cnt ~ ., data = train, method = "anova")
predictions_DT = predict(Dt, test[,-12])


##Random Forest
RF_regressor = randomForest(cnt ~ ., train, importance = TRUE, ntree = 300)
predictions_RF = predict(RF_regressor, test[,-12])
plot(RF_regressor)


#summary(lm_model)

##MAPe
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))*100
}
MAPE(test[,12], predictions_DT)

MAPE(test[,12], predictions_RF)


##extracting predicted value in random forest
results <- data.frame(test, pred_cnt = predictions_RF)

write.csv(results, file = 'RF_R.csv', row.names = FALSE, quote=FALSE)
