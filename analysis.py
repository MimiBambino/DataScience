import pandas
import pandasql
import csv
import numpy as np
import matplotlib.pyplot as plt

filename = ("weather_underground.csv")

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
    print rainy_days
    return rainy_days

num_rainy_days(filename)

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
    print foggy_days
    return foggy_days

max_temp_aggregate_by_fog(filename)

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
    print mean_temp_weekends
    return mean_temp_weekends

avg_weekend_temperature(filename)

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
    print avg_min_temp_rainy
    return avg_min_temp_rainy

avg_min_temperature(filename)

# I don't have the list of filenames for the next 2 functions
def fix_turnstile_data(filenames):
    '''
    Filenames is a list of MTA Subway turnstile text files.
    Returns updated file with only one entry per row.
    '''
    for name in filenames:
        # your code here
        f_in = open(name, 'r')
        f_out = open('updated_'+name, 'w')

        reader_in = csv.reader(f_in, delimiter=',')
        writer_out = csv.writer(f_out, delimiter=',')

        for line in reader_in:
            zero = line[0]
            one = line[1]
            two = line[2]
            num = 3
            for i in range(8):
                try:
                    entry = [zero, one, two, line[num], line[num + 1], line[num + 2], line[num + 3], line[num + 4]]
                    writer_out.writerow(entry)
                    num += 5
                except:
                    continue
        f_in.close()
        f_out.close()

files = ["turnstile_110528.txt", "turnstile_110604.txt"]
fix_turnstile_data(files)
updated_files = ["updated_turnstile_110528.txt", "updated_turnstile_110604.txt"]

def create_master_turnstile_file(filenames, output_file):
    '''
    Consolidates the files in the list filenames into one output_file, including
    a header row.
    '''
    with open(output_file, 'w') as master_file:
       master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')
       for filename in filenames:
            f_in = open(filename, 'r')
            for line in f_in:
                master_file.write(line)
            f_in.close()

create_master_turnstile_file(updated_files, "cynthia_master.csv")

def filter_by_regular(filename):
    '''
    Reads the csv file located at filename into a pandas dataframe, and filters
    the dataframe to only rows where the 'DESCn' column has the value 'REGULAR'.
    '''

    turnstile_data = pandas.read_csv(filename)
    turnstile_data = turnstile_data[turnstile_data['DESCn'] == 'REGULAR']
    return turnstile_data

data = filter_by_regular("cynthia_master.csv")
df = pandas.read(data)

def get_hourly_entries(df):
    '''
    Adds a new column of the count of entries since the last reading
    '''
    df['ENTRIESn_hourly'] = abs(df['ENTRIESn'] - df['ENTRIESn'].shift(1)).fillna(1)
    return df

df = get_hourly_entries(df)

def get_hourly_exits(df):
    '''
    Adds a new column of the count of exits since the last reading
    '''
    #your code here
    df['EXITSn_hourly'] = abs(df['EXITSn'] - df['EXITSn'].shift(1)).fillna(0)
    return df

df = get_hourly_exits(df)

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

def entries_histogram(turnstile_weather):
    '''
    Returns a histogram of entries on rainy days and entries on clear days.
    '''
    rain = turnstile_weather[turnstile_weather['rain'] == 1]
    no_rain = turnstile_weather[turnstile_weather['rain'] == 0]
    plt.figure()
    rain['ENTRIESn_hourly'].hist() # your code here to plot a historgram for hourly entries when it is raining
    no_rain['ENTRIESn_hourly'].hist() # your code here to plot a historgram for hourly entries when it is not raining
    return plt