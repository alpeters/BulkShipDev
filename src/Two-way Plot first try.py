#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt


# In[4]:


# Use Python to read R data files
bulkers_wfr_df = pd.read_csv('/Users/oliver/Desktop/Carbon Emission Project/bulkers_WFR.csv',low_memory=False)
mrv_df = pyreadr.read_r('/Users/oliver/Desktop/Data/MRV.Rda')['MRV_df']
mrv_df = mrv_df.loc[:, ['imo.number', 'reporting.period', 'EU.distance', 'total.fc']]
ais_eu_df = pd.read_csv('/Users/oliver/Desktop/Data/AIS_speed_EEZ_EU_yearly_stats.csv')
ais_eu_df


# In[5]:


mrv_df.columns = ['IMO.Number' if x == 'imo.number' else 
                  'year' if x == 'reporting.period' else
                  x for x in mrv_df.columns]

bulkers_wfr_df.columns = ['mmsi' if x == 'MMSI' else 
                          x for x in bulkers_wfr_df.columns]


# In[6]:


bulkers_wfr_df.dropna(subset=['IMO.Number'], inplace=True)
mrv_df.dropna(subset=['IMO.Number'], inplace=True)

bulkers_wfr_df['IMO.Number'] = bulkers_wfr_df['IMO.Number'].astype(int)
mrv_df['IMO.Number'] = mrv_df['IMO.Number'].astype(int)

# First, join bulkers_wfr_df with mrv_df on 'IMO.Number' and 'Year'
merged_df = pd.merge(bulkers_wfr_df, mrv_df, how='inner', on='IMO.Number')
merged_df['year'] = merged_df['year'].astype('int64')

# Now, merge the resulting df with ais_eu_df on 'MMSI' and 'Year'
final_df = pd.merge(merged_df, ais_eu_df, how='inner', on=['mmsi', 'year'])
final_df['age'] = final_df['year'] - final_df['Built.Year']
final_df['residual'] = np.log1p(final_df['total.fc'].values) - np.log1p(final_df['FC_sum'].values/1000000)
final_df


# In[7]:


final_df[['total.fc','FC_sum']]

# million tonnes or metric tonnes (MRV report)


# In[8]:


tolerance = 500 
final_df['distance_difference'] = abs(final_df['distance_sum'] - final_df['EU.distance'])
final_df = final_df[final_df['distance_difference'] <= tolerance]
final_df


# In[9]:


final_df.to_csv('/Users/oliver/Desktop/Data/df_ml.csv', index=False)


# In[10]:


# Convert pandas series to numpy arrays
total_fc = final_df['total.fc'].values
FC_sum = final_df['FC_sum'].values/1000000

# Fit a line to the data
slope, intercept = np.polyfit(total_fc, FC_sum, 1)

# Generate x values and fitted y values
x_values = total_fc
y_values = slope * x_values + intercept

# Calculate R-squared
r_squared = r2_score(FC_sum, y_values)

# Calculate correlation
correlation = np.corrcoef(total_fc, FC_sum)[0,1]

# Calculate mean squared error
mse = mean_squared_error(FC_sum, y_values)

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(total_fc, FC_sum, label='Data')

# Plot the line of best fit
plt.plot(x_values, y_values, color='red', label='Fit: y = {:.2f}x + {:.2f}'.format(slope, intercept))

# Plot x=y reference line
plt.plot([min(total_fc), max(total_fc)], [min(total_fc), max(total_fc)], 'g--', label='x=y line')

# Set x and y labels
plt.xlabel('Reported Fuel Consumption')
plt.ylabel('Estimated Fuel Consumption')

# Set a title for the plot
plt.title('Comparison of Reported and Estimated Fuel Consumption (m tonnes)')

# Add details in legend about correlation, r-squared and mean squared error
legend_title = 'Fit: y = {:.2f}x + {:.2f}\nCorrelation: {:.2f}\nR²: {:.2f}\nMSE: {:.2f}'.format(slope, intercept, correlation, r_squared, mse)
plt.legend(title=legend_title)

# Show the plot
plt.show()

1.
# systematic reporting bias
# exclude 0 reported fc
# ? source of bias
# subset trips only related to EU to reduce potential bias source


2.
# subset (reported distance sum similar to calculated)
# check if better fit 

# furthur subset the dataset such that all trips will involve EU trip
# exlude ships that have trips outside of EU

# missing hourly data (jump in location of ship) 
# missing observations that are annual power
# possible approach: crude way (divide by hour)
# for instantaneous speed: *implied speed* vs reported speed
# fill in missing value (jump gap) with implied speed, then recalculate everything
# margin of the sample in the 


# In[227]:


# Convert pandas series to numpy arrays and apply log transformation
total_fc = np.log1p(final_df['total.fc'].values)
FC_sum = np.log1p(final_df['FC_sum'].values/1000000)

# Fit a line to the data
slope, intercept = np.polyfit(total_fc, FC_sum, 1)

# Generate x values and fitted y values
x_values = total_fc
y_values = slope * x_values + intercept

# Calculate R-squared
r_squared = r2_score(FC_sum, y_values)

# Calculate correlation
correlation = np.corrcoef(total_fc, FC_sum)[0,1]

# Calculate mean squared error
mse = mean_squared_error(FC_sum, y_values)

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(total_fc, FC_sum, label='Data')

# Plot the line of best fit
plt.plot(x_values, y_values, color='red', label='Fit: y = {:.2f}x + {:.2f}'.format(slope, intercept))

# Plot x=y reference line
plt.plot([min(total_fc), max(total_fc)], [min(total_fc), max(total_fc)], 'g--', label='x=y line')

# Set x and y labels
plt.xlabel('Log Transformed Reported Fuel Consumption')
plt.ylabel('Log Transformed Estimated Fuel Consumption')

# Set a title for the plot
plt.title('Comparison of Log Transformed Reported and Estimated Fuel Consumption (m tonnes)')

# Add details in legend about correlation, r-squared and mean squared error
legend_title = 'Fit: y = {:.2f}x + {:.2f}\nCorrelation: {:.2f}\nR²: {:.2f}\nMSE: {:.2f}'.format(slope, intercept, correlation, r_squared, mse)
plt.legend(title=legend_title)

# Show the plot
plt.show()


3
# exclude 0 and give another try 


4
# add back estimated residuals back to reported fc (two-way plot)


# source of bias brainsotrm
# load variable (instaneous power/reference power)


# derive maximum speed from service speed(service speed is 92% of maximum speed)


# In[224]:


# Convert pandas series to numpy arrays and apply log transformation
total_fc = np.log1p(final_df['total.fc'].values)
FC_sum = np.log1p(final_df['FC_sum'].values/1000000)

r2_score(total_fc, FC_sum)


# In[115]:


# Convert pandas series to numpy arrays and apply log transformation
total_fc = np.log1p(final_df['total.fc'].values)
FC_sum = np.log1p(final_df['time_variant_part_sum'].values)

# Calculate correlation
correlation = np.corrcoef(total_fc, FC_sum)[0,1]
correlation

