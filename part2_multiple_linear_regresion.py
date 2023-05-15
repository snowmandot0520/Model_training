"12.) Multiple linear regression - train and test data"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load the data from the CSV file
data = pd.read_excel("podatki_python_akademija.xlsx")

# Convert 'DATUM' column to datetime format
data['DATUM'] = pd.to_datetime(data['DATUM'], format='%d.%m.%Y')

# Convert 'DATUM' column to string format with desired format
data['DATUM'] = data['DATUM'].dt.strftime('%Y-%m-%d %H:%M:%S')

# split the data into training and testing sets
train_data = data[~data['DATUM'].astype(str).str.startswith('2012-12')]
test_data = data[data['DATUM'].astype(str).str.startswith('2012-12')]

# select the features and target variable
features = ['BDP1', 'BDP2', 'BDP3', 'MIN_VRED_LM1', 'MAX_VRED_LM1']
target = 'MAX_VREDNOST'

# fit a multiple linear regression model on the training set
model = LinearRegression().fit(train_data[features], train_data[target])

# make predictions on the testing set
test_data = test_data.copy()
test_data['PREDICTION'] = model.predict(test_data[features])


# save the predictions and actual values to a CSV file
test_data[['MAX_VREDNOST', 'PREDICTION']].to_csv('predictions.csv', index=False)

# calculate the in-sample and post-sample errors
train_errors_LinearRegression = train_data[target] - model.predict(train_data[features])
test_errors_LinearRegression = test_data[target] - test_data['PREDICTION']

# save the errors to a CSV file
errors_LinearRegression = pd.DataFrame({'In-sample error': train_errors_LinearRegression, 'Post-sample error': test_errors_LinearRegression})
errors_LinearRegression.to_csv('NapakeUcnoTestno.csv.', index=False)

"13.) SVM model"
import pandas as pd
from sklearn.svm import SVR

# create SVM model
model = SVR()

# fit model to training data
model.fit(train_data[features], train_data[target])

# predict on training data and calculate error
train_data = train_data.copy()
train_data.loc[:, 'PREDICTION'] = model.predict(train_data.loc[:, features])
train_errors_SVM = train_data[target] - train_data['PREDICTION']

# predict on test data and calculate error
test_data = test_data.copy()
test_data.loc[:, 'PREDICTION'] = model.predict(test_data.loc[:, features])
test_errors_SVM = test_data[target] - test_data['PREDICTION']

# save the errors to a CSV file
errors_SVM = pd.DataFrame({'In-sample error': train_errors_SVM, 'Post-sample error': test_errors_SVM})
errors_SVM.to_csv('SVM_napake.csv', index=False)

# calculate average post-sample error for each model
avg_error_LinerRegression = errors_LinearRegression['Post-sample error'].mean()
avg_error_SVM = errors_SVM['Post-sample error'].mean()

# print results
if avg_error_LinerRegression < avg_error_SVM:
    print('Linear regression is more accurate')
else:
    print('SVM is more accurate')


"14. How the lag of the training period affects the MAPE error of the model"
import numpy as np

# create empty DataFrame for errors
errors_df = pd.DataFrame(columns=['Training period', 'MAPE error'])

# define start and end dates for training period
train_start = pd.to_datetime('2012-02-01')
train_end = pd.to_datetime('2012-11-30')

# define test period
test_start = pd.to_datetime('2012-12-01')
test_end = pd.to_datetime('2012-12-31')

END = pd.to_datetime('2012-01-07')
data['DATUM'] = pd.to_datetime(data.DATUM)

# loop through training periods with one-day shift
while train_start >= END:
    # create train and test sets
    train_data = data[(data['DATUM'] >= train_start) & (data['DATUM'] <= train_end)]
    test_data = data[(data['DATUM'] >= test_start) & (data['DATUM'] <= test_end)]

    # fit linear regression model and make predictions
    model = LinearRegression().fit(train_data[features], train_data[target])
    test_predictions = model.predict(test_data[features])

    # calculate MAPE error
    mape = np.mean(np.abs((test_data[target] - test_predictions) / test_data[target]))

    # add error to DataFrame
    errors_df = errors_df.append({'Training period': f'{train_start} - {train_end}',
                                  'MAPE error': mape}, ignore_index=True)

    # shift training period by one day
    train_start -= pd.Timedelta(days=1)
    train_end -= pd.Timedelta(days=1)


# print errors
print(errors_SVM)
