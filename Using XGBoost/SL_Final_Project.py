import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import operator
plt.style.use('ggplot')

 
# Helper function for finding left-hand & right-hand average responses (respectively)
def get_avg_responses(x, threshold = 0):

	xc1 = x.loc[data['Left - right handed'] == 0]
	xc2 = x.loc[data['Left - right handed'] == 1]
	resp_avg = pd.concat([xc1.mean(), xc2.mean()], axis = 1)
	resp_avg.columns = ['left handed', 'right handed']
	resp_avg = resp_avg.loc[(resp_avg['left handed'] - resp_avg['right handed']).abs() > threshold]
	
	return resp_avg

# Make a variable binary with respect to some threshold
def make_binary(x, threshold):
	x[x <= threshold] = 0
	x[x > threshold] = 1
	return x
	
# Merges two dictionaries into a new dict as a shallow copy.
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

# A routine for using grid-search to tune "cv_params" while holding "ind_params" constant
# Returns the best score and the best subset of parameters found during the grid search
def tuning_routine(X_train, y_train, ind_params, cv_params):
	
	optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                            scoring = 'accuracy', cv = 5, n_jobs = -1) 
    
	optimized_GBM.fit(X_train, y_train)
	
	return optimized_GBM.best_score_ , optimized_GBM.best_params_
	

# Data exploration: any patterns between left-handed & right-handed young adults?
def perform_DataExploration():
	threshold = 0.1

	## Analyzing average responses for music tastes
	music_avg = get_avg_responses(music_df, threshold)

	music_avg.plot(kind = 'barh', color = ('blue', 'red'))
	plt.xlabel('Average Response: 1 (don\'t enjoy at all) - 5 (enjoy very much)')
	plt.ylabel('Music')
	plt.show()

	## Analyzing average responses for movies
	movies_avg = get_avg_responses(movies_df, threshold)

	movies_avg.plot(kind = 'barh', color = ('blue', 'red'))
	plt.xlabel('Average Response: 1 (don\'t enjoy at all) - 5 (enjoy very much)')
	plt.ylabel('Movies')
	plt.show()

	## Analyzing average responses for hobbies
	hobbies_avg = get_avg_responses(hobbies_df, threshold)

	hobbies_avg.plot(kind = 'barh', color = ('blue', 'red'))
	plt.xlabel('Average Response: 1 (not interested) - 5 (very interested)')
	plt.ylabel('Hobby')
	plt.show()

	## Analyzing average responses for phobias
	phobias_avg = get_avg_responses(phobias_df, threshold)

	phobias_avg.plot(kind = 'barh', color = ('blue', 'red'))
	plt.xlabel('Average Response: 1 (not afraid) - 5 (very afraid)')
	plt.ylabel('Phobia')
	plt.show()

	
# Read in entire dataset into a dataframe
data = pd.read_csv('YP_Dataset.csv', header = 0)

# Encode categorical strings values with integer values
data['Gender'] = data['Gender'].replace(['male', 'female'], [0, 1])

data['Left - right handed'] = data['Left - right handed'].replace(['left handed', 'right handed'], [0, 1])

data['Education'] = data['Education'].replace(['currently a primary school pupil', 'primary school', 'secondary school',
'college/bachelor degree', 'masters degree', 'doctorate degree' ], [0, 1, 2, 3, 4, 5])

data['Only child'] = data['Only child'].replace(['no', 'yes'], [0, 1])

data['Village - town'] = data['Village - town'].replace(['village', 'city'], [0, 1])

data['House - block of flats'] = data['House - block of flats'].replace(['block of flats', 'house/bungalow'], [0, 1])

data['Smoking'] = data['Smoking'].replace(['never smoked', 'tried smoking', 'former smoker', 'current smoker'], [0, 1, 2, 3])

data['Alcohol'] = data['Alcohol'].replace(['never', 'social drinker', 'drink a lot'], [0, 1, 2])

data['Internet usage'] =  data['Internet usage'].replace([ 'no time at all', 'less than an hour a day', 'few hours a day', 'most of the day'], [0, 1, 2, 3])

data['Punctuality'] =  data['Punctuality'].replace(['i am often early','i am always on time', 'i am often running late'], [0, 1, 2])

data['Lying'] =  data['Lying'].replace(['never', 'only to avoid hurting someone', 'sometimes', 'everytime it suits me'], [0, 1, 2, 3])


# Drop rows from dataset in which NaN values occur in the column we want to predict. We don't have a truth value for those.
#y_string =  'Hiphop, Rap'
y_string = 'Hiphop, Rap'
data = data[pd.notnull(data[y_string])]

# Fill in missing values with mean of each column (we want to try to keep as many records as we can)
for column in data:
	data[column] = data[column].fillna(data[column].mean())

# Split dataset into dataframes representing different types of questions
# This makes analyzing a subset of questions easier
music_df = data.ix[:,0:19]     # Music related questions
movies_df = data.ix[:,19:31]   # Movies related questions
hobbies_df = data.ix[:, 31:63] # Hobbies related questions
phobias_df = data.ix[:, 63:73] # Phobias related questions
health_df = data.ix[:, 73:76]  # Health related questions
personality_df = data.ix[:, 76:133]  # Personality related questions
spending_df = data.ix[:, 133:140]    # Spending related questions
demographics_df = data.ix[:, 140::]  # Demographics related questions


#perform_DataExploration()


# 'Action' with threshold 3 has over 10% improvement using all features besides movies
#  Optimal parameters after fine-tuning: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'min_child_weight': 7, 'n_estimators': 80, 'subsample': 0.8, 'reg_lambda': 1, 'objective': 'binary:logistic', 'max_depth': 10, 'gamma': 0}

# 'Hiphop, Rap' with threshold 3 has over 10% improvement using only "Music" features
# Optimal parameters after fine-tuning: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'min_child_weight': 7, 'n_estimators': 80, 'subsample': 1, 'reg_lambda': 10, 'objective': 'binary:logistic', 'max_depth': 5, 'gamma': 0}

if y_string == 'Action':

	# Make prediction variable a binary variable in a way that makes sense
	y = make_binary(movies_df[y_string], 3)
	movies_df = movies_df.drop(y_string, axis = 1)
	
	# Input features matrix to predict 'Action' variable
	X = pd.concat([music_df, hobbies_df, phobias_df, health_df, personality_df, spending_df, demographics_df], axis = 1)
	
	# Final optimal parameters after fine tuning
	final_params = {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'min_child_weight': 7, 'n_estimators': 80, 'subsample': 0.8, 'reg_lambda': 1, 'objective': 'binary:logistic', 'max_depth': 10, 'gamma': 0}
	
	# Best early stopping boosting round found via cross-validation
	best_iteration = 13
	
elif y_string == 'Hiphop, Rap':

	# Make prediction variable a binary variable in a way that makes sense
	y = make_binary(music_df[y_string], 3)
	music_df = music_df.drop(y_string, axis = 1)
	
	# Input features matrix to predict 'Hiphop, Rap' variable
	X = pd.concat([music_df], axis = 1)
	
	# Final optimal parameters after fine tuning
	final_params = {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'min_child_weight': 7, 'n_estimators': 80, 'subsample': 1, 'reg_lambda': 10, 'objective': 'binary:logistic', 'max_depth': 5, 'gamma': 0}

	# Best early stopping boosting round found via cross-validation
	best_iteration = 21

# Split into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 85)

""" 
*** This part of the code is for fine-tuning parameters using sklearn's Grid Search ***

# Parameters sets to be tuned during a 2 pass greedy optimizations
structure_params_opt = {'max_depth': [5, 7, 10], 'min_child_weight' : [5, 6, 7], 'n_estimators': [70, 80, 90]}
learning_params_opt = {'learning_rate': [0.1], 'reg_lambda': [1, 10], 'subsample': [1], 'gamma': [0], 'colsample_bytree': [0.8, 0.9]}
const_params = {'objective': 'binary:logistic'}
learning_params_int = {'learning_rate': 0.1, 'reg_lambda': 1, 'subsample': 0.8, 'gamma': 0, 'colsample_bytree': 0.8}

# First pass: fine-tune associated tree structure parameters
cv_params = structure_params_opt
ind_params = merge_two_dicts(learning_params_int, const_params)
best_score, best_struct_params = tuning_routine(X_train, y_train, ind_params, cv_params)
print "First Tune Best Score: ", best_score

# Second pass: Then use the newly found tree structure parameters and further optimize learning parameters
cv_params = learning_params_opt
ind_params = merge_two_dicts(best_struct_params, const_params)
best_score, best_learning_params = tuning_routine(X_train, y_train, ind_params, cv_params)
print "Second Tune Best Score: ", best_score


# This is the final dictionary of optimized parameters
final_params = merge_two_dicts(ind_params, best_learning_params)
"""

# Create an XGBoost data matrix for the training data. 
# This will be used for early stopping boosting & retrieving feature importance.
xgb_train = xgb.DMatrix(X_train, y_train)

# XGBoost source implementation parameter is called 'eta' (instead of sklearn's wrapper parameter 'learning_rate')
final_params['eta'] = final_params.pop('learning_rate')

"""
*** This part of the code is for finding the optimal early stopping round during boosting using sklearn's Grid Search ***

# The idea is to check to see when validation set accuracy is going down. This means we are overfitting and should stop boosting.
#  The hope is to improve generalization & avoid fitting to the noise of the training data
cv_xgb = xgb.cv(params = final_params, dtrain = xgb_train, num_boost_round = final_params['n_estimators'], nfold = 5,
                metrics = ['error'],
                early_stopping_rounds = 10, verbose_eval = None, show_stdv = False)

# Best numbered round to stop boosting
best_iteration = cv_xgb.tail(1).index[0]
"""


# Train model one last time on all of the optimized parameter, including those found from the grid search & early boosting
final_model = xgb.train(final_params, xgb_train, num_boost_round = best_iteration)

# Create an XGBoost data matrix for the testing data. 
X_test_xgbmat = xgb.DMatrix(X_test)

# With the trained final model, predict the labels of our test set
y_pred = final_model.predict(X_test_xgbmat)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

# Produce accuracy score for the above prediction. The goal is to improve upon the MLE baseline estimate.
accuracy = accuracy_score(y_pred, y_test)


print "Test Set Accuracy: ", accuracy
print "Baseline MLE Accuracy: ", max(1-np.sum(y_train)/float(y_train.size), np.sum(y_train)/float(y_train.size))
importance_dict = final_model.get_score(importance_type= 'gain')


# Select top 'n' feature to show in the importance plot
n = 10
top_n_importance_dict = dict(sorted(importance_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:n])

# Plot top 'n' features with respect to average gain per split
xgb.plot_importance(top_n_importance_dict, xlabel = 'Average Gain Per Split', title = "'" + y_string + "'\n" + 'Top 5 Feature Importance')
plt.show()

