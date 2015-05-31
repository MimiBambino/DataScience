import numpy as np
import pandas
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.

    This can be the same code as in the lesson #3 exercise.
    """

    ###########################
    ### YOUR CODE GOES HERE ###
    ###########################
    lin_model = linear_model.LinearRegression()
    lin_model.fit(features, values)
    params = lin_model.coef_
    intercept = lin_model.intercept_
    #print params[0]
    #print len(lin_model.coef_), len(features)
    return intercept, params

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.

    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe.
    We recommend that you don't use the EXITSn_hourly feature as an input to the
    linear model because we cannot use it as a predictor: we cannot use exits
    counts as a way to predict entry counts.

    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.

    If you receive a "server has encountered an error" message, that means you are
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    # ,Hour,DESCn,ENTRIESn_hourly,EXITSn_hourly,maxpressurei,maxdewpti,mindewpti,minpressurei,meandewpti,meanpressurei,fog,rain,meanwindspdi,mintempi,meantempi,maxtempi,precipi,thunder
    # Select Features (try different features!)
    features = dataframe[['rain']] #.52

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'])
    features = features.join(dummy_units)
    dummy_units = pandas.get_dummies(dataframe['Hour'])
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']

    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    # Perform linear regression
    intercept, params = linear_regression(features_array, values_array)
    #print params
    predictions = intercept + np.dot(features_array, params)
    #plt.scatter(predictions, values)
    #plt.xlabel("Predictions")
    #plt.ylabel("True Values")
    #plt.title("Relationship between Predictions and Values")
    #plt.show()
    return predictions, values

turnstile_weather = 'turnstile_data_master_with_weather.csv'
turnstile_weather = pandas.read_csv(turnstile_weather)

predictions, values = predictions(turnstile_weather)

# plt.scatter(predictions, predictions - values, s=40)
# plt.title("Residuals Plot of Predictions")
# plt.hlines(y = 0, xmin=0, xmax=0)
# plt.ylabel('Residuals')
# plt.show()

plt.hist(predictions - values, bins=100)
plt.title("Residuals Histogram")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.xlim(-10000,10000)
plt.show()