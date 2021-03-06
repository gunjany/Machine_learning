#Polynomial Regression: Bluffing detector

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#Fitting the linear Regression to the datatset
lin_reg = lm(formula = Salary ~ ., data = dataset)

#Fitting the Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2  #adding additional col of level^2 to the dataset
dataset$Level3 = dataset$Level^3  #adding additional col of level^3 to the dataset
dataset$Level4 = dataset$Level^4  #adding additional col of level^4 to the dataset
poly_reg = lm(formula = Salary ~ ., data = dataset)

#Visualising the result of the Linear Regression model
# install.packages('ggplot2')
# library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             colour = 'blue') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), 
            colour = 'red') +
  ggtitle('Truth or Bluff (Linear regression)') +
  xlab('Level') +
  ylab('Salary')
#Visualising the result of the Polynomial Regression model
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             colour = 'blue') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), 
            colour = 'red') +
  ggtitle('Truth or Bluff (Polynomial regression)') +
  xlab('Level') +
  ylab('Salary')
# 
# #Colourful graph for the polynomial regression
# install.packages("esquisse")
# library(esquisse)
# esquisse:: esquisser()
# 
# ggplot(dataset) +
#   geom_point(aes(x = Level, y = Salary, fill = Level2, colour = Level3)) +
#   geom_line(aes(x = Level, y = predict(poly_reg, newdata = dataset), colour = Level3), size =1L ) +
#   scale_fill_viridis_c(option = "plasma") +
#   scale_color_viridis_c(option = "plasma") +
#   labs(x = "Level", y = "Salary", title = "Truth or Bluff", subtitle = "Polynomial Regression") +
#   theme_gray() +
#   theme(legend.position = "none")
# 

# ggplot(dataset) +
#   geom_point(aes(x = Level, y = Salary, fill = Level3, colour = Level4)) +
#   geom_line(aes(x = Level, y = predict(poly_reg, newdata = dataset), colour = Level4), size = 1L) +
#   scale_fill_viridis_c(option = "inferno") +
#   scale_color_viridis_c(option = "inferno") +
#   theme_dark()

#Predicting the new result in Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

#Predicting the new result in Polynomial Regression
y_pred_new = predict(poly_reg, data.frame(Level = 6.5,
                                     Level2 = 6.5^2,
                                     Level3 = 6.5^3,
                                     Level4 = 6.5^4))

#Smoother curve
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
dataset2 = data.frame(Level = x_grid, 
                      Level2 = x_grid^2,
                      Level3 = x_grid^3,
                      Level4 = x_grid^4)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             colour = 'blue') +
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = dataset2) ),
            colour = 'red') +
  ggtitle('Truth or Bluff (Polynomial regression)') +
  xlab('Level') +
  ylab('Salary')



