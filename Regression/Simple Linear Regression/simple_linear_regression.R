
# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#fitting the Simple Linear Regression on the training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#predicting the test set values
y_pred = predict(regressor, test_set)

#Visualising the training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x=training_set$YearsExperience, y= training_set$Salary), 
             colour='red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, training_set)),
            coolour = 'blue')+
  ggtitle('Salary vs Experience (Training Set)')+
  xlab('Years of Experience')+
  ylab('Salary')

#Visualising the Test set results
ggplot() +
  geom_point(aes(x=test_set$YearsExperience, y= test_set$Salary), 
             colour='red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, training_set)),
            coolour = 'blue')+
  ggtitle('Salary vs Experience (Test set)')+
  xlab('Years of Experience')+
  ylab('Salary')

