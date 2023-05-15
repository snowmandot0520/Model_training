import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn import mixture
from sklearn.mixture import GaussianMixture


""" 2.) Load the Excel file into a pandas DataFrame object"""
df = pd.read_excel("podatki_python_akademija.xlsx")


""" 3.), 4.) A linear regression model that predicts MAX_VALUE from MAX_PRET_TED, graph"""
# select the MAX_PRET_TED and MAX_VREDNOST columns for our model
X = df[['MAX_PRET_TED']]
y = df[['MAX_VREDNOST']]

# create a linear regression model
model = LinearRegression().fit(X, y)

# plot the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Regression line')
plt.xlabel('MAX_PRET_TED')
plt.ylabel('MAX_VREDNOST')
plt.show()


""" 6.) A multiple linear regression model that predicts MAX_VALUE from all columns, except: ID, DAN, LETO, DAN_V_TEDNU, TEDEN, DAN_V_MESECU, DATUM, STEVEC and MIN_VREDNOST"""
# select the columns for our model
X = df.drop(columns=['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST'])
y = df[['MAX_VREDNOST']]

# create a linear regression model
model = LinearRegression().fit(X, y)

# calculate the R^2 value, which tells us how well the model fits the data
r_sq = model.score(X, y)
print('R^2 value:', r_sq)

# print the coefficients for each feature
print('Coefficients:', model.coef_)

# print the intercept
print('Intercept:', model.intercept_)

""" 7.) the best Bayesian information multiple linear regression model criterion (BIC)"""

""" I have written the code that works, but names of the variables in .csv are not correct. 
I commented code at the bottom which should work, but I had problems with _get_support_mask() method"""

# Convert datetime_col to to int32
df['DATUM'] = pd.to_datetime(df.DATUM)
# Convert datetime_col to Unix timestamp
df['datetime_col_unix'] = (df['DATUM'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
# Convert datetime_col_unix to int32
df['datetime_col_int32'] = df['datetime_col_unix'].astype(np.int32)

# Drop columns by column names
df = df.drop(['DATUM', 'datetime_col_int32'], axis=1)
X = df.drop(columns=['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'STEVEC', 'MIN_VREDNOST'])


# Initialize variables to store the best BIC and the number of components in the best GMM
best_bic = float('inf')
best_n = None
best_gmm = None

# Iterate over different numbers of components and fit the corresponding GMMs
for n in range(1, len(X.columns) + 1):
    gmm = mixture.GaussianMixture(n_components=n, max_iter=1000, covariance_type='diag', n_init=50)
    gmm.fit(X)
    bic_n = gmm.bic(X)
    if bic_n < best_bic:
        best_bic = bic_n
        best_n = n
        best_gmm = gmm

# Get the names of the variables in the dataset
variable_names = ['var{}'.format(i+1) for i in range(df.shape[1])]

# Get the indices of the components with the largest weight
largest_weight_indices = best_gmm.weights_.argsort()[-1::-1]

# Extract the names and coefficients of the variables in the best model
variable_names_in_best_model = [variable_names[i] for i in best_gmm.means_.argsort(axis=1)[:, largest_weight_indices[0]]]
coefficients_in_best_model = best_gmm.weights_[largest_weight_indices]

# Save the best BIC, the number of components, the names of the variables, and the coefficients to a CSV file
with open('Rezultat.csv', 'w') as f:
    f.write('Best BIC: {}\n'.format(best_bic))
    f.write('Number of components: {}\n'.format(best_n))
    f.write('Variable names: {}\n'.format(','.join(variable_names_in_best_model)))
    f.write('Coefficients: {}\n'.format(','.join(map(str, coefficients_in_best_model))))


""" 10.) MAE, MAPE, SMAPE in RMSE"""
def calculate_errors(df, true_col, pred_col):
    errors = {}
    # Mean Absolute Error
    errors['MAE'] = np.mean(np.abs(df[true_col] - df[pred_col]))
    # Mean Absolute Percentage Error
    errors['MAPE'] = np.mean(np.abs(df[true_col] - df[pred_col]) / np.abs(df[true_col]))
    # Symmetric Mean Absolute Percentage Error
    errors['SMAPE'] = np.mean(2 * np.abs(df[true_col] - df[pred_col]) / (np.abs(df[true_col]) + np.abs(df[pred_col])))
    # Root Mean Squared Error
    errors['RMSE'] = np.sqrt(np.mean((df[true_col] - df[pred_col]) ** 2))
    return errors






