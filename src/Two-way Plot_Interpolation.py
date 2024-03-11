#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pyreadr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Use Python to read R data files
bulkers_wfr_df = pd.read_csv('/Users/oliver/Desktop/Carbon Emission Project/bulkers_WFR.csv',low_memory=False)
mrv_df = pyreadr.read_r('/Users/oliver/Desktop/Data/MRV.Rda')['MRV_df']
mrv_df = mrv_df.loc[:, ['imo.number', 'reporting.period', 'EU.distance', 'total.fc']]
ais_eu_df = pd.read_csv('/Users/oliver/Desktop/Data/AIS_speed_EEZ_EU_yearly_stats_interp.csv')
ais_eu_df


# In[4]:


plt.figure(figsize=(10, 6))
sns.histplot(ais_eu_df['miss_pct_sea_'], kde=True, bins=10)  # Removed the rug=True
#sns.rugplot(ais_eu_df['miss_pct_'], color='black')  # Adding the rugplot separately
plt.title('Distribution of proportion of missing hourly data')
plt.xlabel('miss_pct_sea_')
plt.ylabel('Density')
plt.show()


# In[5]:


plt.figure(figsize=(10, 6))
sns.histplot(ais_eu_df['port_pct_'], kde=True, bins=10)  # Removed the rug=True
#sns.rugplot(ais_eu_df['miss_pct_'], color='black')  # Adding the rugplot separately
plt.title('Distribution of proportion of voyage sitting at the port')
plt.xlabel('port_pct_')
plt.ylabel('Density')
plt.show()


# In[6]:


plt.figure(figsize=(10, 6))
sns.histplot(ais_eu_df['longest_jump_'], kde=True, bins=10)  # Removed the rug=True
#sns.rugplot(ais_eu_df['miss_pct_'], color='black')  # Adding the rugplot separately
plt.title('Distribution of longest jump')
plt.xlabel('longest_jump_')
plt.ylabel('Density')
plt.show()


# In[7]:


total_obs = len(ais_eu_df['miss_pct_sea_'])

above_0_75 = len(ais_eu_df[ais_eu_df['miss_pct_sea_'] > 0.75]) / total_obs * 100
between_0_50_0_75 = len(ais_eu_df[(ais_eu_df['miss_pct_sea_'] <= 0.75) & (ais_eu_df['miss_pct_sea_'] > 0.50)]) / total_obs * 100
between_0_25_0_50 = len(ais_eu_df[(ais_eu_df['miss_pct_sea_'] <= 0.50) & (ais_eu_df['miss_pct_sea_'] > 0.25)]) / total_obs * 100
below_0_25 = len(ais_eu_df[ais_eu_df['miss_pct_sea_'] <= 0.25]) / total_obs * 100

print(f"Above 0.75: {above_0_75:.2f}%")
print(f"Between 0.50 and 0.75: {between_0_50_0_75:.2f}%")
print(f"Between 0.25 and 0.50: {between_0_25_0_50:.2f}%")
print(f"Below 0.25: {below_0_25:.2f}%")


# In[7]:


mrv_df.columns = ['IMO.Number' if x == 'imo.number' else 
                  'year' if x == 'reporting.period' else
                  x for x in mrv_df.columns]

bulkers_wfr_df.columns = ['mmsi' if x == 'MMSI' else 
                          x for x in bulkers_wfr_df.columns]


# In[8]:


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


# In[9]:


final_df[['total.fc','FC_sum']]

# million tonnes or metric tonnes (MRV report)


# In[10]:


tolerance = 500 
final_df['distance_difference'] = abs(final_df['distance_sum'] - final_df['EU.distance'])
final_df = final_df[final_df['distance_difference'] <= tolerance]
final_df


# In[21]:


final_df.to_csv('/Users/oliver/Desktop/Data/df_interp_ml.csv', index=False)


# In[26]:


total_obs = len(final_df['miss_pct_'])

above_0_75 = len(final_df[final_df['miss_pct_'] > 0.75]) / total_obs * 100
between_0_50_0_75 = len(final_df[(final_df['miss_pct_'] <= 0.75) & (final_df['miss_pct_'] > 0.50)]) / total_obs * 100
between_0_25_0_50 = len(final_df[(final_df['miss_pct_'] <= 0.50) & (final_df['miss_pct_'] > 0.25)]) / total_obs * 100
below_0_25 = len(final_df[final_df['miss_pct_'] <= 0.25]) / total_obs * 100

print(f"Above 0.75: {above_0_75:.2f}%")
print(f"Between 0.50 and 0.75: {between_0_50_0_75:.2f}%")
print(f"Between 0.25 and 0.50: {between_0_25_0_50:.2f}%")
print(f"Below 0.25: {below_0_25:.2f}%")


# In[23]:


final_df_before = pd.read_csv('/Users/oliver/Desktop/Data/df_ml.csv')


# In[14]:


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


# In[15]:


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


# In[33]:


# Convert pandas series to numpy arrays
total_fc_interp = final_df['total.fc'].values
FC_sum_interp = final_df['FC_sum'].values/1000000

slope1, intercept1 = np.polyfit(total_fc_interp, FC_sum_interp, 1)

# Generate x values and fitted y values
x_values1 = total_fc_interp
y_values1 = slope1 * x_values1 + intercept1

total_fc2 = final_df_before['total.fc'].values
FC_sum2 = final_df_before['FC_sum'].values/1000000

slope2, intercept2 = np.polyfit(total_fc2, FC_sum2, 1)
x_values2 = total_fc2
y_values2 = slope2 * x_values2 + intercept2

# Create a scatter plot
plt.figure(figsize=(10, 8))

# --- First Plot ---
plt.scatter(total_fc_interp, FC_sum_interp, label='Interpolation')
plt.plot(x_values1, y_values1, color='red', label=f'Fit 1: y = {slope1:.2f}x + {intercept1:.2f}')

# --- Second Plot ---
plt.scatter(total_fc2, FC_sum2, label='Original', alpha=0.5)
plt.plot(x_values2, y_values2, color='blue', label=f'Fit 2: y = {slope2:.2f}x + {intercept2:.2f}')

# Common plot settings
plt.plot([min(min(total_fc_interp), min(total_fc2)), max(max(total_fc_interp), max(total_fc2))],
         [min(min(total_fc_interp), min(total_fc2)), max(max(total_fc_interp), max(total_fc2))], 'g--', label='x=y line')

plt.xlabel('Reported Fuel Consumption')
plt.ylabel('Estimated Fuel Consumption')
plt.title('Comparison of Reported and Estimated Fuel Consumption (m tonnes)')

# Calculate R-squared and correlation for both plots
r_squared1 = r2_score(FC_sum_interp, y_values1)
correlation1 = np.corrcoef(total_fc_interp, FC_sum_interp)[0,1]
r_squared2 = r2_score(FC_sum2, y_values2)
correlation2 = np.corrcoef(total_fc2, FC_sum2)[0,1]

# Display R-squared and correlation in legend
legend_title = (
    f'Fit 1: y = {slope1:.2f}x + {intercept1:.2f}, R²: {r_squared1:.2f}, Correlation: {correlation1:.2f}\n'
    f'Fit 2: y = {slope2:.2f}x + {intercept2:.2f}, R²: {r_squared2:.2f}, Correlation: {correlation2:.2f}'
)
plt.legend(title=legend_title, loc='upper left')

# Show the plot
plt.show()


# In[34]:


# Convert pandas series to numpy arrays
total_fc_interp = np.log1p(final_df['total.fc'].values)
FC_sum_interp = np.log1p(final_df['FC_sum'].values/1000000)

slope1, intercept1 = np.polyfit(total_fc_interp, FC_sum_interp, 1)

# Generate x values and fitted y values
x_values1 = total_fc_interp
y_values1 = slope1 * x_values1 + intercept1

total_fc2 = np.log1p(final_df_before['total.fc'].values)
FC_sum2 = np.log1p(final_df_before['FC_sum'].values/1000000)

slope2, intercept2 = np.polyfit(total_fc2, FC_sum2, 1)
x_values2 = total_fc2
y_values2 = slope2 * x_values2 + intercept2

# Create a scatter plot
plt.figure(figsize=(10, 8))

# --- First Plot ---
plt.scatter(total_fc_interp, FC_sum_interp, label='Interpolation')
plt.plot(x_values1, y_values1, color='red', label=f'Fit 1: y = {slope1:.2f}x + {intercept1:.2f}')

# --- Second Plot ---
plt.scatter(total_fc2, FC_sum2, label='Original', alpha=0.5)
plt.plot(x_values2, y_values2, color='blue', label=f'Fit 2: y = {slope2:.2f}x + {intercept2:.2f}')

# Common plot settings
plt.plot([min(min(total_fc_interp), min(total_fc2)), max(max(total_fc_interp), max(total_fc2))],
         [min(min(total_fc_interp), min(total_fc2)), max(max(total_fc_interp), max(total_fc2))], 'g--', label='x=y line')

plt.xlabel('Reported Fuel Consumption')
plt.ylabel('Estimated Fuel Consumption')
plt.title('Comparison of Reported and Estimated Fuel Consumption (m tonnes)')

# Calculate R-squared and correlation for both plots
r_squared1 = r2_score(FC_sum_interp, y_values1)
correlation1 = np.corrcoef(total_fc_interp, FC_sum_interp)[0,1]
r_squared2 = r2_score(FC_sum2, y_values2)
correlation2 = np.corrcoef(total_fc2, FC_sum2)[0,1]

# Display R-squared and correlation in legend
legend_title = (
    f'Fit 1: y = {slope1:.2f}x + {intercept1:.2f}, R²: {r_squared1:.2f}, Correlation: {correlation1:.2f}\n'
    f'Fit 2: y = {slope2:.2f}x + {intercept2:.2f}, R²: {r_squared2:.2f}, Correlation: {correlation2:.2f}'
)
plt.legend(title=legend_title, loc='upper left')

# Show the plot
plt.show()


# In[35]:


final_df


# In[36]:


final_df_before

# which voyage ; most EU
# subset ships where 80% of trips are within EU
# interpolation 6hr gap instead 1hr


# In[ ]:


# longest distance between two points (each ship annual level) another feature
# percentage of voyage thats sitting at the port
# subset ships where operational phase is sea: only used to generate features to offset bias(dataset 1: conditioning ships is sailing in the sea): use this to compute missing percnetage 
# let jasper run on multiple dataset
# Port phase: any activity report with a speed of less than 3 knots


# In[ ]:


2.114

