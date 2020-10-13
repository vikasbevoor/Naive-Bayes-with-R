#Import the raw_sms dataset
sms_raw = read.csv("D:/Data science/Assignments docs/Naive Bayes/sms_raw.csv")
View(sms_raw)

# Examining the imported data
colnames((sms_raw))
str(sms_raw)

# Changing the "type" as factor
sms_raw$type <- factor(sms_raw$type)
sms_raw$type

# Examining the type variable
table(sms_raw$type)
prop.table(table(sms_raw$type)) * 100

# Barplot of sms type
tbl <- table(sms_raw$type)
barplot(tbl, beside = TRUE, legend = TRUE)

library(tm)
# transforming the SMS text into a corpus
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# cleaning up the corpus using tm_map()
corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_dtm

# creating training and test datasets
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test  <- sms_raw[4170:5559, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5559]

# check that the proportion of spam is similar
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

# dictionary of words which are used more than 5 times
sms_dict <- findFreqTerms(sms_dtm_train, 5)
View(sms_dict)

sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))
sms_test

# Defining function to convert counts to a factor: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)
View(sms_test)

library(e1071)
#  Training a model on the data
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
View(sms_classifier)

#  Evaluating model performance
sms_test_pred <- predict(sms_classifier, sms_test)
View(sms_test_pred)

# Table of types of predicted values 
table(sms_test_pred)
prop.table(table(sms_test_pred))

library(gmodels)
# table of actual v/s predicted
CrossTable(sms_test_pred, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

# Confusion matrix 
library(caret)
confusionMatrix(sms_test_pred, sms_raw_test$type)
