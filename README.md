# Adding-Domain-Knowledge-to-Keras-based-Neural-Networks

## Methods
1. Residual modelling
2. Domain knowledge as a regularizer (using custom loss function)

### Residual modelling
Training the neural network to predict the error(residual) of domain model predictions, and correcting the  domain prediction using the error predictions.

*residual = y_domain-y_true*

*y_pred =y_domain - residual_pred*

### Domain knowledge as a regularizer
Using a custom loss functions, training the neural network to reduce the error between predictions and true values as well as the error between predictions and domain model predictions.
custom loss = RMS(ypred  - ytrue) + r.RMS(ydomain- ypred)
where r controls the impact of the domain model predictions.

## Datasets
1. API metrics
2. API Manager
3. TPC-W 
4. Springboot 
5. Springboot
6. Ballerina
7. API Manager

For each dataset 4 types of models are trained to compare results:
1. Machine Learning Model (Neural Network)
2. Domain Model (curve fitting for USL)
3. Residual Model
4. Machine Learning Model with Custom Loss (domain model as a regularizer for neural networks)

## Extended residual modelling
Training the residual model by replacing the domain model with a second machine learning model. 

Machine learning models used:
1. XGBoost
2. Linear Regression
