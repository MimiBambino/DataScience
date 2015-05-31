#Analyzing the NYC Subway Dataset
by Cynthia O'Donnell

##References
- [SciPy Documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu)
- [Mann-Whitney U Test Wiki](http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
- [Stack Overflow](http://stackoverflow.com/questions/20095673/python-shift-column-in-pandas-dataframe-up-by-one)
- [Greg Reda on Pandas data structures](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/)
- [Stack Exchange](http://stats.stackexchange.com/questions/31361/some-questions-about-two-sample-comparisons)
- [The Miniab Blog](http://blog.minitab.com/blog/adventures-in-statistics/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit)
- [Big Data Examiner](http://www.bigdataexaminer.com/how-to-run-linear-regression-in-python-scikit-learn/)

##Section 1. Statistical Test
*1.1 Which statistical test did you use to analyse the NYC subway data? Did you use a one-tail or a two-tail P value? What is the null hypothesis? What is your p-critical value?*

> This analysis determines whether ridership on the New York City Subway system is significantly different on rainy days versus on clear/non-rainy days. The null hypothesis is that rain has no effect on mean ridership, i.e., the means of the rainy day ridership and clear day ridership are equal.  Therefore, this is a two-sided problem, because we have not hypothesized that one sample will be higher than the other, just that they are not equal. In order to test the hypothesis, I used the Mann-Whitney U Test.  This test provides a one-sided p-value, so we need to multiply the p-value returned by the test by 2.  The p-critical value is 5%.

*1.2 Why is this statistical test applicable to the dataset? In particular, consider the assumptions that the test is making about the distribution of ridership in the two samples.*

> This data is not normally distributed and many other statistical tests such as Welch's t-test assume normally distributed data. The Mann-Whitney U Test does not assume normal distribution.  The Mann-Whitney U Test does assume a certain minimum sample size (20 for each sample), but this data is sufficiently large to use this test.

*1.3 What results did you get from this statistical test? These should include the following numerical values: p-values, as well as the means for each of the two samples under test.*
> ```
Mean entries with rain: 1105.4463767458733
Mean entries without rain: 1090.278780151855
U-statistic: 1924409167.0
p-value: 0.024999912793489721
```

*1.4 What is the significance and interpretation of these results?*

> The p-critical value is .05.  As stated above, we must multiply the test statistic by 2 because it is a two sided test. So the test statistic of 0.04999982558 just barely beats the critical value and we can reject the null hypothesis. The U-statistic is very large. The maximum U value is the product of the two sample sizes and our U-statistic is very near that.  A U statistic that is very close to the maximum value increases my confidence that this result is statistically significant despite the fact that the p value is so close to the p critical value.

##Section 2. Linear Regression
*2.1 What approach did you use to compute the coefficients theta and produce prediction for ENTRIESn_hourly in your regression model?*

> I used the linear_model from the sklearn module with the default parameter values.

*2.2 What features (input variables) did you use in your model? Did you use any dummy variables as part of your features?*

> I used rain, UNIT as a dummy variable and Hour as another dummy variable.

*2.3 Why did you select these features in your model? We are looking for specific reasons that lead you to believe that the selected features will contribute to the predictive power of your model.*

> I started the analysis with several of the variables and UNIT as a dummy variable. After quite a bit of experimentation, I found that adding Hour as a dummy variable improved R2 by about 9%. I included Hour because I assumed that ridership would be influenced by the time of day, especially if it was also raining.

>After that, I began removing variables to see how much R2 would decrease. I found that removing some variables did not affect R2 at all.  Removing all of the others only affected R2 by 0.2% or less.  Eventually, I just had rain.  I kept rain instead of another highly correlated variable (like precipi) because it was more relevant to the study.

>Finally, I included the UNITs as dummy variables, because this drastically improved R2.

*2.4 What are the coefficients (or weights) of the non-dummy features in your linear regression model?*

> `[78.4591959117]`

*2.5 What is your model’s R2 (coefficients of determination) value?*

> `0.520444206356`

*2.6 What does this R2 value mean for the goodness of fit for your regression model? Do you think this linear model to predict ridership is appropriate for this dataset, given this R2 value?*

> R2 is the percentage of variance that is explained by the model.  Higher is generally better, and this model's R2 is higher than 0.4. However, when determining goodness of fit, it is important to consider a residual plot as well.

> A residual plot will help determine whether the model is biased in an obvious way.  In particular if there is a discernable pattern in the residuals plot, the model is biased.

![Residuals plot](residualslink)

> This residuals plot indicates that there are trends that the model is not picking up.  As a result, I do not believe that this model fits the data well enough.  I feel that a different type of model might be a better choice for this data.

##Section 3. Visualization
*3.1 Include and describe a visualization containing two histograms: one of  ENTRIESn_hourly for rainy days and one of ENTRIESn_hourly for non-rainy days.*

![ENTRIESn_hourly Histogram](entrieshourlyhistogram-link)

> This plot shows the number of entries per hour on rainy days and the number of entries per hour on clear days on the same graph.  This data is clearly not normally distributed, but the two distributions are strikingly similar to each other.

> There are twice as many clear days than rainy days which is not apparent from this graph.  As this is a histogram, it takes the cumulative number of entries on rainy days and on clear days.  With twice as many days, it is not surprising that there are nearly twice as many cumulative entries on the clear days as on the rainy days.  Understanding this, it is telling that none of the clear day bins is twice as high as the rainy day bins.  This, rather unintuitively, indicates that proportional ridership is actually greater on rainy days, which supports our rejection of the null hypothesis.

*3.2 Include and describe a freeform visualization.*

![Average Number of Riders By Hour](entriesbytime-link)

> I plotted the average number of riders per hour. There are many peaks and valleys.  The true ridership is surely more even than this graph indicates.  The data for New York City subway ridership reports the number of entries every 4 hours with a few other oddly timed reports.  Therefore, there are peaks at 8:00am, noon, 4:00pm and 8:00pm.  It might be more useful to use a smoothing algorithm to show an average ridership, but plotting data points that we don't actually have might also be misleading.

> From this plot it is clear that there are many more riders in the afternoon than in the morning which is interesting. If we assume that most of the riders are commuting to and from work, it would appear that they generally work later than the traditional 9:00 to 5:00 work hours.  This might be because subway commuters are working in retail, restaurants or some other business that operates later in the day.  More data is necessary to determine if that is true.  Additionally, New York City is a major tourist destination and perhaps tourists tend to use the subway and also begin their sightseeing in the late morning and into the night.

##Section 4. Conclusion
*4.1 From your analysis and interpretation of the data, do more people ride the NYC subway when it is raining or when it is not raining?*

> Particularly given the results from the Mann-Whitney U test (p-value: 0.025), we can say with a high level of certainty that more people ride the NYC subway when it is raining.  It is important to note that simply looking at the means of both data sets is insufficient, due to variance.  The Mann-Whitney U test is needed to quantitatively confirm that the two data sets are statistically different.

*4.2 What analyses lead you to this conclusion? You should use results from both your statistical tests and your linear regression to support your analysis.*

> The positive coefficient for the rain (0 or 1) parameter indicates that the presence of rain contributes to increased ridership.  This may have not been the case for all data points, with the R^2 being approximately 46%; however, the small residuals show relatively high accuracy, given our objectives.  Although the means of both data sets are not that different from each other, the Mann-Whitney U test did indicate that there was a statistically significant change in ridership for rain vs. no-rain.  It is conscientious to claim that rain increases subway ridership.

##Section 5. Reflection
*5.1 Please discuss potential shortcomings of the methods of your analysis, including: data set, linear regression model, and statistical tests.*

> My initial critique of the study is that the dataset is too small.  This dataset is comprised of data only for the month of May.  It is troubling to me to try to abstract any conclusions regarding this dataset to any other month.  Even if we are just trying to predict how ridership is impacted by rain in May, we should have May data from other years in order to reach a more tenable conclusion.

> If we look at the data as a sample of rainy days and clear days, the sample size is actually 10 for the rainy days and 20 for the clear days. The Mann-Whitney U-test requires a sample size of at least 20 and, therefore, we should not use the Mann-Whitney U-test for this data because the sample of rainy days is not large enough.

> It would be nice to have data regarding demographics of subway riders. What proportion of subway riders are tourists, how many people are monthly passholders, what are the occupations of the riders, how many riders own vehicles, and whether they are recreational riders or commuters. The proportion of tourists in May and the proportion of recreational riders are important because, when it rains, recreational riders may choose to just stay indoors, thereby reducing ridership overall.  But commuters may be more likely to ride.

> If data from other months was included we should broaden the hypothesis to precipitation as opposed to rain because several months of the year it will not rain, but snow.  This information should be captured in the precipitation variable.

> I am not convinced that linear regression is the best model choice.  Especially if more data were included, a classification model such as a decision tree might better predict the ridership with precipitation.  Particularly, as the mean temperature rises and falls throughout the year, precipitation may have a larger impact on ridership.

*5.2 Do you have any other insight about the dataset that you would like to share with us?*

> It is an interesting dataset.  While it has a lot of data points, it is quite obviously too small to make any broad generalizations regarding the hypothesis that rain impacts ridership.  Additionally, it is unfortunate that the weather data merely indicates whether or not it rained on a given day.  It is rare that it rains for an entire day or exactly from midnight to midnight.  If it rained overnight and not during the day, then the data could indicate 2 rainy days when the overnight rain had no impact at all on peak ridership from 11:00am to 10:00pm.  Hourly rain data is needed for a more granular analysis of the impact of rain on ridership.
