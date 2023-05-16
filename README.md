# Data_analysis_training
This repository contains exercises and code related to a training in data analysis using the pandas library in Python. The exercises cover various topics such as reading and processing data, creating linear regression and SVM models, calculating errors, working with different file formats and implementing unit tests.

All the code is in three different files:
-  `part1_BIC.py`
-  `part2_multiple_linear_regresion.py`
-  `part3_creating_json.py`

## Exercises

The following is a list of exercises covered in the training:

1\. **Reading and exploring .xlsx files**: Learn about the .xlsx file format and explore how to read such data into a Python program.

2\. **Reading data using pandas**: Apply the knowledge gained and read data from the "podatki_python_akademija.xlsx" document into a Python program using the pandas library.

3\. **Linear regression model**: Learn about linear regression and create a linear regression model that predicts MAX_VREDNOST based on MAX_PRET_TED using pandas.

4\. **Plotting regression line**: Visualize the regression line for the linear regression model created in the previous exercise.

5\. **Difference between linear regression and multiple linear regression**: Investigate the differences between linear regression and multiple linear regression.

6\. **Multiple linear regression model**: Create a multiple linear regression model that predicts MAX_VREDNOST using all columns except specific ones.

7\. **Best model selection using BIC**: Create the best multiple linear regression model using the Bayesian Information Criterion (BIC) and determine the coefficients of the selected variables. Save the variable names and their coefficients in the "Rezultat.csv" file.

8\. **Error calculations**: Learn about the definitions of MAE, MAPE, SMAPE, and RMSE errors and calculate these errors for the previously built model.

9\. **Function creation**: Learn about functions in Python and create a function that takes a DataFrame and the names of two columns (measured values column and calculated values column) as input. The function should return a list of calculated errors: MAE, MAPE, SMAPE, and RMSE.

10\. **Naming components of a list**: Investigate how Python names components of a list and update the function created in the previous exercise to have appropriate component names.

11\. **Adding function values to a file**: Add the values calculated by the function in Exercise 9, with their corresponding names, to the "Rezultat.csv" file.

12\. **Splitting data and creating a new model**: Split the data into a training period (January to November) and a test period (December). Create the best multiple linear regression model based on the BIC criterion using the training data. Search online for a function that predicts MAX_VREDNOST for the test period based on the model created using the training period. Calculate the errors (MAE, MAPE, SMAPE, and RMSE) for the in-sample (training period) and post-sample (test period) and save the results in the "NapakeUcnoTestno.csv" file.

13\. **Support Vector Machine (SVM) model**: Use the training and test data from the previous exercise and create an SVM model. Calculate the errors (MAE, MAPE, SMAPE, and RMSE) for the training and test periods and save the results in the "SVM_napake.csv" file. Compare the performance of the SVM model with the linear regression model.

14\. **Effect of training period shift on MAPE error**: Study how shifting the training period affects the MAPE error of the model. Select the data for the training period from February 1st to November 30th and keep the test period from December 1st to December 31st. Calculate the MAPE error for the test period. Shift the training data back by one day (January 31st to November 29th) while keeping the test period unchanged, and calculate the MAPE error again. Repeat this process by shifting the training data by one day until you reach the first data point of the training period (January 7th to November 5th). Create a function to define the boundaries of the training period using a for loop and counters to iterate over the dates.

15\. **Working with .json files**: Learn about the .json file format and explore how to read and create such files in Python.

16\. **Converting .csv to .json**: Utilize the knowledge gained to read the .csv file from Exercise 2 and output it as a .json file named "one_line_data.json".

17\. **Converting .json to .csv**: Read the "one_line_data.json" file and convert it to a .csv file named "one_line_data.csv". Output the resulting .csv file.

18\. **Unit testing**: Learn about unit tests and how to use them in Python programs.

19\. **Unit testing on .csv files**: Apply the knowledge of unit tests and create a successful unit test that compares the .csv file from Exercise 2 with the newly created .csv file from Exercise 17.
