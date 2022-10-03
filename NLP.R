library("spacyr")
library(dplyr)
library(tidytext)
library(stringr)
#############################################################################################
#        Data Upload
#############################################################################################
dataset <- read.csv('train.csv')
#############################################################################################
#        Data View
#############################################################################################
sum(is.na(dataset))
str(dataset)
head(dataset)

#############################################################################################
#        Data Pre-processing (Features generation)
#############################################################################################
# to generate new feature holding number of word count
dataset$nWords <- sapply(strsplit(dataset$text, " "), length)
# to generate new feature holding number of letters count
dataset$nChar <- nchar(dataset$text)  
# number of characters (doesn't count spaces)
dataset$nCharNoSpace <- dataset$nChar - (dataset$nWords -1)
# average number of characters in a word in sentence
dataset$aveLetters <- dataset$nCharNoSpace/dataset$nWords     

#universal part of speech tag set
types <- c ('ADJ','ADP','ADV','AUX','CCONJ','DET',             
            'INTJ','NOUN','NUM','PART','PRON','PROPN',
            'PUNCT','SCONJ','SYM','VERB','X') 
typesNorm <- paste0(types,'2')

# to tokenize and tag the text, and returns a data table of the results
hrr <- data.frame(spacy_parse(dataset$text, tag = TRUE, pos = TRUE)) 

# to generate a frequency table for all words in the dataset
stststs2 <- data_frame(text = hrr$lemma) %>% 
  unnest_tokens(word, text) %>%    # split words
  #anti_join(stop_words) %>%    # take out "a", "an", "the", etc.
  count(word, sort = TRUE)    # count occurrences

#newwrrrr <- data.frame(spacy_parse(dataset$text[1], tag = TRUE, pos = TRUE))

# to extract the most frequent (1100) words used in the dataset
words <- stststs2$word[1:1100]
# Note: most frequent 1100 words are taken to be attributes for each observation 
#       this number can be change or even to take all words ("words 1-gram")


# to visualize most frequent words in dataset
library(wordcloud)
wordcloud(stststs2$word, stststs2$n, max.words=100, random.order=FALSE, colors = rainbow(30))
wordcloud(words, scale = c(2, 1), min.freq = 50, colors = rainbow(30))
wordcloud(words = stststs2$word, freq = stststs2$n, min.freq = 1,
          max.words=100, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))

# to generate new attributes holding count of different tags and most frequent words for each observation
for (rw in 1:nrow(dataset)){
  newwr <- data.frame(spacy_parse(dataset$text[rw], tag = TRUE, pos = TRUE))
  for (pos in types){
    dataset[rw, pos] <- sum(newwr$pos==pos)
    dataset[rw, paste0(pos,'2')] <- dataset[rw, pos]/dataset[rw, 'nWords']
  }
  for (pos in words){
    dataset[rw, pos] <- sum(tolower(newwr$lemma)==pos)
  }
}
# to check number of features after new features generation
dim(dataset)

#############################################################################################
#        Data Pre-processing
#############################################################################################

#write.csv(numData, "numData.csv", row.names = FALSE)
#write.csv(datasetNor, "datasetNor.csv", row.names = FALSE)
#write.csv(datasetNorL, "datasetNorL.csv", row.names = FALSE)

# to drop unimportant features ("id", "text")
numData <- dataset[-c(1,2)]
# to factor output feature ("author")
numData$author <- as.factor(numData$author)

# to convert all data to numeric values, so it can be normalized
numData$author <- unclass(numData$author)
numData <- data.frame(sapply(numData, function(x) as.numeric(x)))

# to check correlation between all features
x <- cor(as.matrix(numData), method = c("pearson", "kendall", "spearman"))

# Normalizing function
normalize <- function (x){
  value = (x-min(x))/(max(x)-min(x))
  if (max(x) == min(x)){ ## To eliminate error if max(x)==min(x). I have one attribute such that.
    value = 0
  }
  return(value)
}
testeer <- as.data.frame(numData[,apply(numData, 2, var, na.rm=TRUE) != 0]) # to eliminate attributes with zero variation; (max()==min())

# to normalize data features values
datasetNor <- as.data.frame(lapply(numData[,],normalize))

which(is.na(datasetNor))
datasetNorL <- datasetNor
datasetNorL$author <- dataset$author
str(datasetNorL)#Check factors

# to factor output feature
datasetNorL$author <- factor(datasetNorL$author)

# to split data into training and testing subsets
library(caTools)
set.seed(123)
split <- sample.split(datasetNor$author, SplitRatio = 0.80)
train_set <- subset(datasetNor, split == TRUE)
test_set = subset(datasetNor, split == FALSE)
train_lables <- train_set[1]
test_lables <- test_set[1]
train_set1 <- subset(datasetNorL, split == TRUE)
test_set1 <- subset(datasetNorL, split == FALSE)
train_lables1 <- train_set1[,1]
test_lables1 <- test_set1[,1]
# to check representation of the output factors in output feature subsets against all dataset
prop.table(table(datasetNorL$author))
prop.table(table(train_set1$author))
prop.table(table(test_set1$author))
hist(datasetNor$author) 

#############################################################################################
#        Data training and testing
#############################################################################################

####################    Knn    #########################

library(class)
library(gmodels)
Knn_test_pred1 <- knn(train = train_set1[,-1], test = test_set1[,-1], cl = train_lables1, k = 5)
CrossTable(x = test_lables1, y = Knn_test_pred1, prop.chisq = FALSE)
confusionMatrix(Knn_test_pred1, test_lables1)

Knn_test_pred2 <- knn(train = train_set1[,-1], test = test_set1[,-1], cl = train_lables1, k = 25)
CrossTable(x = test_lables1, y = Knn_test_pred2, prop.chisq = FALSE)
confusionMatrix(Knn_test_pred2, test_lables1)

Knn_test_pred3 <- knn(train = train_set1[,-1], test = test_set1[,-1], cl = train_lables1, k = 39)
CrossTable(x = test_lables1, y = Knn_test_pred3, prop.chisq = FALSE)
confusionMatrix(Knn_test_pred3, test_lables1)

Knn_test_pred4 <- knn(train = train_set1[,-1], test = test_set1[,-1], cl = train_lables1, k = 125) 
CrossTable(x = test_lables1, y = Knn_test_pred4, prop.chisq = FALSE)


Knn_test_pred5 <- knn(train = train_set1[,-1], test = test_set1[,-1], cl = train_lables1, k = 1)
CrossTable(x = test_lables1, y = Knn_test_pred5, prop.chisq = FALSE)
confusionMatrix(Knn_test_pred5, test_lables1)

####################   Random Forest   #########################

library(randomForest)
RF_classifier2 = randomForest(author ~ . ,
                            data = train_set1,
                            ntree = 300, importance = TRUE)
RF_test_pred2 = predict(RF_classifier1, newdata = test_set1[-1])
CrossTable(test_lables1, RF_test_pred2, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)
confusionMatrix(RF_test_pred2, test_lables1)




####################   naive bayes    #########################

library(e1071)
NB_classfier1 <- naiveBayes(train_set1, train_set1$author)
NB_test_pred1 <- predict(NB_classfier1, test_set1)
length(test_lables[,1])
CrossTable(x = test_lables1, y = NB_test_pred1, prop.chisq = FALSE)
CrossTable(NB_test_pred1, test_lables1, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, prop.c = FALSE, dnn = c('predicted', 'actual'))
confusionMatrix(NB_test_pred1, test_lables1)

####################   Regression    #########################
Rg_classifier1 <- lm(author ~ ., data = train_set)
Rg_test_pred1 <- predict(Rg_classifier1, test_set)
cor(Rg_test_pred1, test_set$author)
range(Rg_test_pred1)
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}
MAE(test_set$author, Rg_test_pred1)
####################   Neural Networks    #########################
library(neuralnet)
set.seed(12345)
NN_classfier1 <- neuralnet(formula = author ~ ., data = train_set1)
plot(NN_classfier1)
NN_test_result1 <- compute(NN_classfier1, test_set1[-1])
NN_test_result2 <- predict(NN_classfier1, test_set1[-1])
write.csv(NN_test_result2, "NN_test_result2.csv", row.names = FALSE)
str(NN_test_result1)
table(test_set1$author, NN_test_result2)

NN_test_pred1 <- NN_test_result2$net.result
CrossTable(NN_test_pred1, test_set1$author)
helpere <- as.data.frame(NN_test_result2)
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}
MAE(test_set$author, NN_test_pred1)
#Try different hidden layers and nodes NN_classfier2 <- neuralnet(author ~ ., data = train_set, hidden = c(3,3), learningrate = 0.01)

######################       Support Vector Machine    #########################
library(kernlab)
library(caret)
SVM_classfier1 <- ksvm(author ~ ., data = train_set1, kernel = "vanilladot")
SVM_test_pred1 <- predict(SVM_classfier1, test_set1)
table(SVM_test_pred1, test_set1$author)#cor ot MAE
confusionMatrix(SVM_test_pred1, test_set1$author)

## Model Improving by changing kernel ##

SVM_classfier2 <- ksvm(author ~ ., data = train_set1, kernel = "rbfdot") #the best among all kernels
SVM_test_pred2 <- predict(SVM_classfier2, test_set1)
confusionMatrix(SVM_test_pred2, test_set1$author)

SVM_classfier3 <- ksvm(author ~ ., data = train_set1, kernel = "polydot")
SVM_test_pred3 <- predict(SVM_classfier3, test_set1)
confusionMatrix(SVM_test_pred3, test_set1$author)

SVM_classfier4 <- ksvm(author ~ ., data = train_set1, kernel = "laplacedot")
SVM_test_pred4 <- predict(SVM_classfier4, test_set1)
confusionMatrix(SVM_test_pred4, test_set1$author)

SVM_classfier5 <- ksvm(author ~ ., data = train_set1, kernel = "anovadot")#Bad
SVM_test_pred5 <- predict(SVM_classfier5, test_set1)
confusionMatrix(SVM_test_pred5, test_set1$author)

SVM_classfier6 <- ksvm(author ~ ., data = train_set1, kernel = "stringdot")# Bad
SVM_test_pred6 <- predict(SVM_classfier6, test_set1)
confusionMatrix(SVM_test_pred6, test_set1$author)

SVM_classfier9 <- ksvm(author ~ ., data = train_set1, kernel = "tanhdot")
SVM_test_pred9 <- predict(SVM_classfier9, test_set1)
confusionMatrix(SVM_test_pred9, test_set1$author)

SVM_classfier0 <- ksvm(author ~ ., data = train_set1, kernel = "besseldot")
SVM_test_pred0 <- predict(SVM_classfier0, test_set1)
confusionMatrix(SVM_test_pred0, test_set1$author)
###########

## Model Improving by changing type ##

# did not try type = "C-svc" because it is the defualt.
SVM_classfier7 <- ksvm(author ~ ., data = train_set1, type = "spoc-svc", kernel = "rbfdot") # the best
SVM_test_pred7 <- predict(SVM_classfier7, test_set1)
confusionMatrix(SVM_test_pred7, test_set1$author)

SVM_classfier8 <- ksvm(author ~ ., data = train_set1, type = "kbb-svc", kernel = "rbfdot")
SVM_test_pred8 <- predict(SVM_classfier8, test_set1)
confusionMatrix(SVM_test_pred8, test_set1$author)

?ksvm
#########################
 
## Model Improving by introducing 5 fold cross validation and change C parameter to 5##
# C is cost of constraints violation 
set.seed(12345)
SVM_classfier10<-ksvm(author ~ ., data = train_set1, type = "spoc-svc", kernel = "rbfdot", cross = 5, C = 5)
SVM_test_pred10 <- predict(SVM_classfier10, test_set1)
confusionMatrix(SVM_test_pred10, test_set1$author) # it has the best accuracy with 77.98%

## Model Improving by tuning parameters ##
control2 <- trainControl(method="repeatedcv", number=2, repeats=2, 
                         classProbs = TRUE, summaryFunction =  defaultSummary,
                         search = "grid")

require(kernlab)
SVM_classfier14 <- train(author ~ ., data = train_set1,
                         method = "svmRadial",  
                         trControl = control2,
                         metric = "Accuracy", tuneGrid = expand.grid(C = seq(40, 60, length = 4), sigma = c(0.001,0.05)))

#SVM_classfier12 <- train(author ~ ., data = train_set1,
#                    method = "svmRadial",  
#                    trControl = control2,
#                    metric = "Accuracy ", tuneGrid = expand.grid(C = seq(0, 50, length = 10)))
#Error: The tuning parameter grid should have columns sigma, C
print(SVM_classfier12)
plot(SVM_classfier12)

control3 <- trainControl(method="repeatedcv", number=2, repeats=5, 
                         classProbs = TRUE, summaryFunction =  defaultSummary,
                         search = "grid")

SVM_classfier15 <- train(author ~ ., data = train_set1,   
                         method = "svmRadial",  
                         trControl = control2,
                         metric = "Accuracy", 
                         tuneGrid = expand.grid(C = 5,sigma = seq(0.0001, 0.1, length = 5)))

print(SVM_classfier15)
plot(SVM_classfier15)
#grid <- expand.grid(sigma = c(.01, .015, 0.2), C = c(0.75, 0.9, 1, 1.1, 1.25))

## Till now SVM is the best with accuracy of 77.89%
