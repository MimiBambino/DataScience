import pandas
import pandasql
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from ggplot import *
import sys
import statsmodels.api as sm

def num_rainy_days(filename):
    '''
    Run a SQL query on a dataframe of weather data.  Returns the number of days
    it rained.
    '''

    weather_data = pandas.read_csv(filename)

    q = """
    SELECT count(rain) FROM weather_data WHERE rain = 1;
    """

    #Execute your SQL command against the pandas frame
    rainy_days = pandasql.sqldf(q.lower(), locals())
    print "Rainy days: ", rainy_days
    return rainy_days

#num_rainy_days('weather_underground.csv')

def max_temp_aggregate_by_fog(filename):
    '''
    Run a SQL query on a dataframe of weather data.  Return the maximum temperature
    for both foggy and non-foggy days.
    '''
    weather_data = pandas.read_csv(filename)

    q = """
    SELECT fog, max(maxtempi)
    FROM weather_data
    GROUP BY fog;
    """

    #Execute your SQL command against the pandas frame
    foggy_days = pandasql.sqldf(q.lower(), locals())
    print "Foggy days: ", foggy_days
    return foggy_days

def avg_weekend_temperature(filename):
    '''
    Run a SQL query on a dataframe of weather data.  Return the average mean
    temperature on weekends.
    '''
    weather_data = pandas.read_csv(filename)

    q = """
    SELECT avg(meantempi)
    FROM weather_data
    WHERE cast(strftime('%w', date) as integer) = 6 or cast(strftime('%w', date) as integer) = 0;
    """

    #Execute your SQL command against the pandas frame
    mean_temp_weekends = pandasql.sqldf(q.lower(), locals())
    print "Mean temp weekends: ", mean_temp_weekends
    return mean_temp_weekends

def avg_min_temperature(filename):
    '''
    Run a SQL query on a dataframe of weather data. Return the average minimum
    temperature on rainy days where the minimum temperature is greater than 55 degrees.
    '''
    weather_data = pandas.read_csv(filename)

    q = """
    SELECT avg(mintempi)
    FROM weather_data
    WHERE rain = 1 and mintempi > 55;
    """

    #Execute your SQL command against the pandas frame
    avg_min_temp_rainy = pandasql.sqldf(q.lower(), locals())
    print "Avg_min_temp_rainy: ", avg_min_temp_rainy
    return avg_min_temp_rainy

# def fix_turnstile_data(filenames):
#     '''
#     Filenames is a list of MTA Subway turnstile text files.
#     Returns updated file with only one entry per row.
#     '''
#     for name in filenames:
#         # your code here
#         f_in = open(name, 'r')
#         f_out = open('updated_'+name, 'w')

#         reader_in = csv.reader(f_in, delimiter=',')
#         writer_out = csv.writer(f_out, delimiter=',')

#         for line in reader_in:
#             zero = line[0]
#             one = line[1]
#             two = line[2]
#             num = 3
#             for i in range(8):
#                 try:
#                     entry = [zero, one, two, line[num], line[num + 1], line[num + 2], line[num + 3], line[num + 4]]
#                     writer_out.writerow(entry)
#                     num += 5
#                 except:
#                     continue
#         f_in.close()
#         f_out.close()

# files =
# fix_turnstile_data(files)
# updated_files = ["updated_turnstile_110528.txt", "updated_turnstile_110604.txt"]

# def create_master_turnstile_file(filenames, output_file):
#     '''
#     Consolidates the files in the list filenames into one output_file, including
#     a header row.
#     '''
#     with open(output_file, 'w') as master_file:
#        master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
#        for filename in filenames:
#             f_in = open(filename, 'r')
#             for line in f_in:
#                 master_file.write(line)
#             f_in.close()

# create_master_turnstile_file(updated_files, "cynthia_master.csv")

# def filter_by_regular(filename):
#     '''
#     Reads the csv file located at filename into a pandas dataframe, and filters
#     the dataframe to only rows where the 'DESCn' column has the value 'REGULAR'.
#     '''

#     turnstile_data = pandas.read_csv(filename)
#     turnstile_data = turnstile_data[turnstile_data['DESCn'] == 'REGULAR']
#     return turnstile_data

# data = filter_by_regular("cynthia_master.csv")

# def get_hourly_entries(df):
#     '''
#     Adds a new column of the count of entries since the last reading
#     '''
#     df['ENTRIESn_hourly'] = abs(df['ENTRIESn'] - df['ENTRIESn'].shift(1)).fillna(1)
#     return df

# df = get_hourly_entries(data)

# def get_hourly_exits(df):
#     '''
#     Adds a new column of the count of exits since the last reading
#     '''
#     #your code here
#     df['EXITSn_hourly'] = abs(df['EXITSn'] - df['EXITSn'].shift(1)).fillna(0)
#     return df

# df = get_hourly_exits(df)

def time_to_hour(time):
    '''
    Separate a time string in the format of: "00:00:00" (hour:minutes:seconds)
    Return the integer of the hour. For example:
        1) if hour is 00, your code should return 0
        2) if hour is 01, your code should return 1
        3) if hour is 21, your code should return 21

    Please return hour as an integer.
    '''

    hour = time[:2]
    if hour[0] == "0":
        hour = int(hour[1])
    else:
        hour = int(hour)
    return hour

def reformat_subway_dates(date):
    '''
    Write a function that takes as its input a date in the the format
    month-day-year, and returns a date in the format year-month-day.
    '''

    date_formatted = "20" + date[6:] + "-" + date[:5]
    return date_formatted

turnstile_weather = "turnstile_data_master_with_weather.csv"
turnstile_weather = pandas.read_csv(turnstile_weather)

def entries_histogram(turnstile_weather):
    '''
    Returns a histogram of entries on rainy days and entries on clear days.
    '''
    no_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 0]
    rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 1]

    bins = 400
    alpha = 0.7
    xmin = ymin = 0
    xmax = 5000
    ymax = 35000

    plt.figure()
    no_rain.hist(bins=bins, alpha=alpha) # your code here to plot a historgram for hourly entries when it is not raining
    rain.hist(bins=bins, alpha=alpha) # your code here to plot a historgram for hourly entries when it is raining

    plt.axis([xmin, xmax, ymin, ymax])
    plt.suptitle('Histogram of ENTRIESn_hourly')
    plt.xlabel('ENTRIESn_hourly')
    plt.ylabel('Count')
    plt.legend(['No rain', 'Rain'])
    plt.show()

    return plt

entries_histogram(turnstile_weather)

def riders_by_hour(turnstile_weather):

    hourly_avg = []
    for i in range(23):
        data = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['Hour'] == i]
        avg = np.mean(data)
        hourly_avg.append(avg)

    plt.plot(range(23), hourly_avg, linewidth=4, c='g')
    plt.xlim(0,22)
    plt.xlabel('Hour')
    plt.title('Average Number of Riders Per Hour')
    plt.ylabel('Rider Count')
    plt.fill_between(range(23), hourly_avg, color='green')

    plt.show()

#riders_by_hour(turnstile_weather)

def mann_whitney_plus_means(turnstile_weather):
    '''
    Calculates the means of the entries with rain and without rain and runs the
    Mann Whitney U-test.
    '''

    ### YOUR CODE HERE ###
    rain = turnstile_weather[turnstile_weather['rain'] == 1]
    no_rain = turnstile_weather[turnstile_weather['rain'] == 0]
    with_rain_mean = np.mean(rain['ENTRIESn_hourly'])
    without_rain_mean = np.mean(no_rain['ENTRIESn_hourly'])
    U, p = scipy.stats.mannwhitneyu(rain['ENTRIESn_hourly'],no_rain['ENTRIESn_hourly'])

    print  with_rain_mean, without_rain_mean, U, p
    return with_rain_mean, without_rain_mean, U, p # leave this line for the grader

#mann_whitney_plus_means(turnstile_weather)

# # Gradient Descent Linear Model
# def normalize_features(df):
#     """
#     Normalize the features in the data set.
#     """
#     mu = df.mean()
#     sigma = df.std()

#     if (sigma == 0).any():
#         raise Exception("One or more features had the same value for all samples, and thus could " + \
#                          "not be normalized. Please do not include features with only a single value " + \
#                          "in your model.")
#     df_normalized = (df - df.mean()) / df.std()

#     return df_normalized, mu, sigma

# def compute_cost(features, values, theta):
#     """
#     Compute the cost function given a set of features / values,
#     and the values for our thetas.
#     """
#     # your code here
#     m = len(values)
#     SSE = np.square(np.dot(features, theta) - values).sum()
#     cost = SSE/(2*m)
#     return cost

# def gradient_descent(features, values, theta, alpha, num_iterations):
#     """
#     Perform gradient descent given a data set with an arbitrary number of features.
#     """

#     m = len(values)
#     cost_history = []

#     for i in range(num_iterations):
#         # your code here
#         predicted_values = np.dot(features, theta)
#         theta = theta - alpha / m * np.dot((predicted_values - values), features)

#         cost = compute_cost(features, values, theta)
#         cost_history.append(cost)
#     return theta, pandas.Series(cost_history)

# def predictions(dataframe):
#     '''
#     If you are using your own algorithm/models, see if you can optimize your code so
#     that it runs faster.
#     '''
#     # Select Features (try different features!)
#     #features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]     #0.46397
#     #features = dataframe[['rain', 'precipi', 'Hour', 'mintempi']]      #0.46429
#     #features = dataframe[['rain', 'meanwindspdi', 'Hour', 'mintempi']] #0.46467
#     #features = dataframe[['rain', 'meanwindspdi', 'Hour', 'mintempi', 'fog']] #0.46536, num_iter = 100
#     #features = dataframe[['rain', 'meanwindspdi', 'Hour', 'mintempi', 'fog']] #0.46532, alpha=0.05
#     features = dataframe[['rain', 'meanwindspdi', 'Hour', 'mintempi', 'fog']]

#     # Add UNIT to features using dummy variables
#     dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
#     features = features.join(dummy_units)

#     # Values
#     values = dataframe['ENTRIESn_hourly']
#     m = len(values)

#     features, mu, sigma = normalize_features(features)
#     features['ones'] = np.ones(m) # Add a column of 1s (y intercept)

#     # Convert features and values to numpy arrays
#     features_array = np.array(features)
#     values_array = np.array(values)

#     # Set values for alpha, number of iterations.
#     alpha = 0.2 # please feel free to change this value
#     num_iterations = 100 # please feel free to change this value

#     # Initialize theta, perform gradient descent
#     theta_gradient_descent = np.zeros(len(features.columns))
#     theta_gradient_descent, cost_history = gradient_descent(features_array,
#                                                             values_array,
#                                                             theta_gradient_descent,
#                                                             alpha,
#                                                             num_iterations)

#     plot = None
#     plot = plot_cost_history(alpha, cost_history)

#     predictions = np.dot(features_array, theta_gradient_descent)
#     return predictions, plot


# def plot_cost_history(alpha, cost_history):
#    """This function is for viewing the plot of your cost history.
#    You can run it by uncommenting this

#        plot_cost_history(alpha, cost_history)

#    call in predictions.

#    If you want to run this locally, you should print the return value
#    from this function.
#    """
#    cost_df = pandas.DataFrame({
#       'Cost_History': cost_history,
#       'Iteration': range(len(cost_history))
#    })
#    return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
#       geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )

def plot_residuals(turnstile_weather, predictions):
    '''
    Plot a histogram of the residuals (the difference between the original
    hourly entry data and the predicted values).
    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    plt.figure()
    (turnstile_weather['''ENTRIESn_hourly'''] - predictions).hist(bins=100)
    return plt

# def compute_r_squared(data, predictions):
#     '''
#     Given a list of data pointpredicted data points, calculate the R^2 value.
#     '''

#     # your code here
#     SST = ((data - np.mean(data))**2).sum()
#     SSReg = ((predictions - data)**2).sum()
#     r_squared = 1 - SSReg / SST
#     return r_squared

# def normalize_features(df):
#     """
#     Normalize the features in the data set.
#     """
#     mu = df.mean()
#     sigma = df.std()

#     if (sigma == 0).any():
#         raise Exception("One or more features had the same value for all samples, and thus could " + \
#                          "not be normalized. Please do not include features with only a single value " + \
#                          "in your model.")
#     df_normalized = (df - df.mean()) / df.std()

#     return df_normalized, mu, sigma

# def predictions(weather_turnstile):
#     #
#     # Your implementation goes here. Feel free to write additional
#     # helper functions
#     #
#     features = weather_turnstile[['rain', 'meanwindspdi', 'Hour', 'mintempi', 'fog']]

#     # Add UNIT to features using dummy variables
#     dummy_units = pandas.get_dummies(weather_turnstile['UNIT'], prefix='unit')
#     features = features.join(dummy_units)

#     values = weather_turnstile['ENTRIESn_hourly']
#     m = len(values)

#     features, mu, sigma = normalize_features(features)
#     features['ones'] = np.ones(m) # Add a column of 1s (y intercept)

#     features_array = np.array(features)
#     features_array = sm.add_constant(features_array)

#     values_array = np.array(values)

#     model = sm.OLS(values_array, features_array)
#     res = model.fit()
#     prediction = model.predict(res.params)
#     return prediction

# if __name__ == '__main__':
    # print "Number of rainy days:"
    # print num_rainy_days('weather_underground.csv')
    # raw_input("Press Enter to continue...")
    # print "Maximum temperature aggregate by fog:"
    # print max_temp_aggregate_by_fog('weather_underground.csv')
    # raw_input("Press Enter to continue...")
    # print "Average minimum temperature:"
    # print avg_min_temperature('weather_underground.csv')
    # raw_input("Press Enter to continue...")
    # print "Fixed turnstile data:"
    # fix_turnstile_data(["turnstile_110528.txt", "turnstile_110604.txt"])
    # print open('updated_turnstile_110507.txt').read()
    # raw_input("Press Enter to continue...")
    # print "Filter by regular:"
    # df = filter_by_regular('turnstile_data.csv')
    # print df
    # raw_input("Press Enter to continue...")
    # print "Hourly entries:"
    # df = pandas.read_csv('turnstile_data.csv')
    # print get_hourly_entries(df)
    # raw_input("Press Enter to continue...")