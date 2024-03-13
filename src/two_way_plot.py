"""
Create two-way plot between (log) reported and calculated fuel consumption

"""

import os 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

datapath = 'src/data'
final_df = pd.read_csv(os.path.join(datapath, 'df_ml.csv'))


# Two-way plot between reported and calculated FC

total_fc = final_df['report_fc']
FC_sum = final_df['cal_fc']

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


# Two-way plot between log-transformed reported and calculated FC

total_fc = final_df['log_report_fc']
FC_sum = final_df['log_cal_fc']


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