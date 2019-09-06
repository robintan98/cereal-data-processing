###Robin Tan############################################################################################################
###Wharton Analytics Fellows Data Challenge#############################################################################
###Spring 2019##########################################################################################################

###Imports##############################################################################################################
import pandas as pd
import random
from sklearn import linear_model
from sklearn.metrics import r2_score

###Parameters###########################################################################################################
num_of_data = 69
num_of_train = 53
num_of_test = num_of_data - num_of_train
num_of_factors = 9
num_of_trials = 100
total_r_squared = 0
total_adjusted_r_squared = 0

###Model################################################################################################################
all_cereal_data = pd.read_csv(r'C:\Users\Robin\Documents\Robin\Penn\Processed Cereal Data.csv')
factors_data = all_cereal_data[['Calories','Protein','Fat','Sodium','Fiber','Carbo','Sugars','Potass','Vitamins']]
ratings_data = all_cereal_data[['Rating']]

for i in range(num_of_trials):
    randomize_indices = list(range(0, num_of_data))
    random.shuffle(randomize_indices)
    train_indices = randomize_indices[0:num_of_train]
    test_indices = randomize_indices[num_of_train:num_of_data]

    factors_data_train = factors_data.iloc[train_indices].values.reshape(-1, num_of_factors)
    factors_data_test = factors_data.iloc[test_indices].values.reshape(-1, num_of_factors)

    ratings_data_train = ratings_data.iloc[train_indices].values.reshape(-1, 1)
    ratings_data_test = ratings_data.iloc[test_indices].values.reshape(-1, 1)

    lrm = linear_model.LinearRegression()
    model = lrm.fit(factors_data_train, ratings_data_train)
    predicted_ratings = model.predict(factors_data_test)

    r_squared = r2_score(ratings_data_test, predicted_ratings)
    adjusted_r_squared = 1 - (1 - r_squared)*(num_of_test - 1)/(num_of_test - num_of_factors - 1)
    total_r_squared += r_squared
    total_adjusted_r_squared += adjusted_r_squared

average_r_squared = total_r_squared / num_of_trials
average_adjusted_r_squared = total_adjusted_r_squared / num_of_trials

###Results##############################################################################################################
print ('Average R^2: ' + str(average_r_squared))
print('Average Adjusted R^2: ' + str(average_adjusted_r_squared))

###Prediction###########################################################################################################
all_data_indices = list(range(0,num_of_data))
prediction_indices = list([69, 70, 71, 72, 73])
factors_all_data_train = factors_data.iloc[all_data_indices].values.reshape(-1, num_of_factors)
factors_all_data_test = factors_data.iloc[prediction_indices].values.reshape(-1, num_of_factors)
ratings_all_data_train = ratings_data.iloc[all_data_indices].values.reshape(-1, 1)

new_lrm = linear_model.LinearRegression()
new_model = new_lrm.fit(factors_all_data_train, ratings_all_data_train)
new_predicted_ratings = new_model.predict(factors_all_data_test)

print('Model coefficients: ' + str(new_model.coef_))
print('Model intercept: ' + str(new_model.intercept_))
print('Predicted ratings: ' + str(new_predicted_ratings))
