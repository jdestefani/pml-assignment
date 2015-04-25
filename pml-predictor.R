# Library includes
library(caret)
library(doParallel)
library(randomForest)

# Auxiliary functions
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


# Preprocessing training data
# Temporal data needs to be separated into homogeneous chunks.
setwd("~/workspace/OnlineCourses//PracticalMachineLearning");
rawTrainingData <- read.csv("pml-training.csv");
testData <- read.csv("pml-testing.csv");

# Find all the columns containing missing values (i.e. Na, "", "#DIV/0!")
missingValues <- sapply(rawTrainingData, function(x){x=="" || x=="#DIV/0!" || is.na(x)})
rawTrainingData[missingValues] <- NA

qplot(trainingData$user_name,trainingData$X,colour=trainingData$classe)
qplot(trainingData$X,trainingData$classe)

# Remove all the columns that have a majority of missing values

# Automated way - Verify whether a column of the data frame contains at least a NA value
sapply(trainingData, function(x){anyNA(x, recursive = FALSE)})

# Deterministic way
missingValuesColumns <- ( grepl(glob2rx("^kurtosis*"),names(rawTrainingData)) 
                          | grepl(glob2rx("^skewness*"),names(rawTrainingData)) 
                          | grepl(glob2rx("^kurtosis*"),names(rawTrainingData))
                          | grepl(glob2rx("^avg*"),names(rawTrainingData))
                          | grepl(glob2rx("^var*"),names(rawTrainingData))
                          | grepl(glob2rx("^stddev*"),names(rawTrainingData))
                          | grepl(glob2rx("^max*"),names(rawTrainingData))
                          | grepl(glob2rx("^min*"),names(rawTrainingData))
                          | grepl(glob2rx("^amplitude*"),names(rawTrainingData)))

trainingData <- rawTrainingData[,!missingValuesColumns]

#Remove meaningless columns for the analysis ?
trainingData <- trainingData[,8:dim(trainingData)[2]]

# Verify highly correlated predictors and verify whether or not to remove them from the dataset
# Possibly include them in the analysis
numericVarCorrelationMatrix <- cor(trainingData[,sapply(trainingData,function(x){class(x) == "numeric" || class(x) == "integer" })])
highlyCorrelatedPairs <- which(numericVarCorrelationMatrix > 0.7,arr.ind=T)
#highlyCorrelatedPairs[,1] <- names(rawTrainingData)[highlyCorrelatedPairs[,1]]
#highlyCorrelatedPairs[,2] <- names(rawTrainingData)[highlyCorrelatedPairs[,2]]


# Keep all the rows having complete data - 406 obs ~ 2% of the total number of observations
#completeRows <- trainingData[trainingData$new_window == "yes",]

# MODEL FITTING
# Split training-test set - Random Forest
inTrainingF <- createDataPartition(y=trainingData$classe,p=.6,list=F)
trainingF <- trainingData[inTrainingF,]
testingF <- trainingData[-inTrainingF,]

set.seed(17711771)

## Set up parallel processing - Random forest generation
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

basicForest <- randomForest(trainingF$classe~., data=trainingF,ntree=50,importance=T,proximity=T,oob.prox=T,allowParallel=T)

## turn off parallel processing
stopCluster(cluster)

predictions <- predict(basicForest,newdata=testing)
results <- confusionMatrix(predictions,testing$classe)
accuracy <- results$overall[1]


# Split training-test set - LDA & Naive Bayes
inTrainingN <- createDataPartition(y=trainingData$classe,p=.8,list=F)
trainingN <- trainingData[inTrainingN,]
testingN <- trainingData[-inTrainingN,]

set.seed(17711771)

# 10-fold CV
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

#gbmFit1 <- train(Class ~ ., data = training,
#                 method = "gbm",
#                 trControl = fitControl,
#                 ## This last option is actually one
#                 ## for gbm() that passes through
#                 verbose = FALSE)

## Set up parallel processing - LDA & Naive Bayes
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

modelLDA <- train(trainingN$classe~., data=trainingN,method="lda",allowParallel=T)
modelNB <- train(trainingN$classe~., data=trainingN,method="nb",allowParallel=T)

## turn off parallel processing
stopCluster(cluster)

predictionsLDA <- predict(modelLDA,newdata=testingN)
resultsLDA <- confusionMatrix(predictionsLDA,testingN$classe)

predictionsNB <- predict(modelNB,newdata=testingN)
resultsNB <- confusionMatrix(predictionsNB,testingN$classe)
