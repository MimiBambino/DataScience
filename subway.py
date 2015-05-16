# borrowed from http://nbviewer.ipython.org/url/www.alma.cl/~itoledo/Presentation1.ipynb

import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from ggplot import *

import plotly.plotly as py
import plotly.tools as tls
from plotly.graph_objs import *

pd.options.display.max_columns = 50
pd.options.mode.chained_assignment = None

improved_df = pd.read_csv(
    'turnstile_weather_v2.csv')
improved_df['dateTime'] = improved_df.apply(
    lambda r: pd.datetime.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S'), axis=1)
improved_df = improved_df.set_index('dateTime', drop=False)
improved_df['hour_week'] = improved_df.day_week * 24. + improved_df.hour
improved_df['day'] = improved_df.index.day
improved_df['week'] = improved_df.index.week
print "The number of entries in the data set is: %d" % len(improved_df)
print "The number of turnstiles is: %d" % len(improved_df.UNIT.unique())
