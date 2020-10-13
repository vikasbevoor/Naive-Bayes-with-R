#Import the salary dataset 
Sal <- read.csv("D:/Data science videos/R Codes/Assignments docs/Naive Bayes/SalaryData_Train (2).csv")
View(Sal)
library(readr)

# Data exploration
summary(Sal)
str(Sal)

attach(Sal)
levels(Sal$Salary)

# examine the type variable more carefully
table(Salary)  
str(Salary)

# Visualizations
ggplot(Sal, aes(age, colour = Salary)) +
  geom_freqpoly(binwidth = 1) + labs(title="Age Distribution by Salary")

c <- ggplot(Sal, aes(x=hoursperweek, fill=Salary, color=Salary)) +
  geom_histogram(binwidth = 1) + labs(title="Hours per week Distribution by Salary")
c + theme_bw()


# creating training and test datasets
sal_train <- Sal
sal_test  <- SalT

# check that the proportion of Salary is similar
prop.table(table(Sal$Salary))

install.packages("e1071")
library(e1071)
library(caret)

##  Training a model on the data ----
sal_classifier <- naiveBayes(Salary~., data =sal_train)
sal_classifier

##  Evaluating model performance ----
sal_test_pred <- predict(sal_classifier, sal_test, type = "class")
sal_test_pred
str(sal_test_pred)

# Accuracy
table(sal_test_pred)
prop.table(table(sal_test_pred))
confusionMatrix(table(sal_test_pred,sal_test$Salary))

library(gmodels)
CrossTable(sal_test_pred, sal_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))