"""
Calculate ship-specific fixed component of power from WFR data
Input(s): bulkers_WFR.csv
Output(s): bulkers_WFR_calcs.csv
Runtime:
"""
# Reference IMO4 in annotations refers to the Fourth IMO GHG Study (Faber et al. 2020)

#%%
import re
import os
import numpy as np
import pandas as pd
import pyreadr # for reading RData files
import seaborn as sns # plots for publication
import matplotlib.pyplot as plt # for plotting

datapath = 'data'
plotpath = 'plots'
plotprefix = 'WFR_Calculations_Fixed_Power_'

# Functions definitions
def categorize_engine(row):
    """
    Parse 'Main.Engine.Detail' to extract engine features and then categorize the engine type as per IMO procedure (Faber et al. 2020)

    Args:
        row: A row of the WFR dataframe with 'Main.Engine.Detail' column.

    Returns:
        Series: Engine type, RPM, and reference power (mkW).
    """
    # Extract features from 'Main Engine Detail' column
    detail = row['Main.Engine.Detail']
    fuel_type = str(row['Main.Engine.Fuel.Type']).lower() if row['Main.Engine.Fuel.Type'] is not None else ''
    
    # Regular expression patterns
    patterns = {
        'rpm': r'at ([\d,]+\.?\d*)rpm',
        'model_name': r'Diesel - (.*?) -',
        'num_stroke': r'(\d+)-stroke',
        'mkW': r'(\d+(?:,\d+)?)mkW',
    }
    
    # Extract and format features
    features = {k: re.search(v, detail) for k, v in patterns.items()}
    rpm = float(features['rpm'].group(1).replace(',', '')) if features['rpm'] else np.nan
    model_name = features['model_name'].group(1).lower() if features['model_name'] else ''
    num_stroke = int(features['num_stroke'].group(1)) if features['num_stroke'] else np.nan
    mkW = float(features['mkW'].group(1).replace(',', '')) if features['mkW'] else np.nan

    # Initialize 'category' to np.nan
    category = np.nan

    # Engine categorization
    if "lng" in fuel_type:
        if "wartsila" in model_name or "wingd" in model_name:
            category = 'LNG-Otto SS'
        elif num_stroke == 4 and rpm > 300 and 'lbsi' not in model_name:
            category = 'LNG-Otto MS'
        elif "man energy solutions" in model_name or "man b. & w." in model_name:
            category = 'LNG-Diesel'
        elif 'lbsi' in model_name:
            category = 'LBSI'
    elif 'mdo' in fuel_type or 'ifo' in fuel_type or 'biofuel' in fuel_type:
        if rpm <= 300:
            category = 'SSD'
        elif 300 < rpm <= 900:
            category = 'MSD'
        elif rpm > 900:
            category = 'HSD'
    
    # Assign the extracted mkW to a new column 'ME_W_ref'
    row['ME_W_ref'] = mkW
    
    return pd.Series([category, rpm, mkW])


def assign_sfc_base(row):
    """
    Assigns base SFC values from Table 19 of IMO4

    Args:
        row: A row of the WFR dataframe with 'Engine_Category', 'Main.Engine.Fuel.Type', and 'Built.Year' columns.

    Returns:
        Tuple: Base SFC values for main engine, auxiliary engine, and boiler.
    """

    engine_category = row['Engine_Category']
    fuel_type = str(row['Main.Engine.Fuel.Type']).lower() if row['Main.Engine.Fuel.Type'] is not None else ''
    built_year = row['Built.Year']
    
    # ME_sfc_base
    ME_sfc_base = None
    # SSD category
    if engine_category == 'SSD':
        if 'ifo' in fuel_type:
            if built_year < 1983:
                ME_sfc_base = 205
            elif 1984 <= built_year <= 2000:
                ME_sfc_base = 185
            else:
                ME_sfc_base = 175
        elif 'mdo' in fuel_type or 'biofuel' in fuel_type:
            if built_year < 1983:
                ME_sfc_base = 190
            elif 1984 <= built_year <= 2000:
                ME_sfc_base = 175
            else:
                ME_sfc_base = 165

    # MSD category
    elif engine_category == 'MSD':
        if 'ifo' in fuel_type:
            if built_year < 1983:
                ME_sfc_base = 215
            elif 1984 <= built_year <= 2000:
                ME_sfc_base = 195
            else:
                ME_sfc_base = 185
        elif 'mdo' in fuel_type or 'biofuel' in fuel_type:
            if built_year < 1983:
                ME_sfc_base = 200
            elif 1984 <= built_year <= 2000:
                ME_sfc_base = 185
            else:
                ME_sfc_base = 175

    # HSD category
    elif engine_category == 'HSD':
        if 'ifo' in fuel_type:
            if built_year < 1983:
                ME_sfc_base = 225
            elif 1984 <= built_year <= 2000:
                ME_sfc_base = 205
            else:
                ME_sfc_base = 195
        elif 'mdo' in fuel_type or 'biofuel' in fuel_type:
            if built_year < 1983:
                ME_sfc_base = 210
            elif 1984 <= built_year <= 2000:
                ME_sfc_base = 190
            else:
                ME_sfc_base = 185

    # LNG engines
    elif "lng" in fuel_type:
        if engine_category == 'LNG-Otto MS':
            if 1984 <= built_year <= 2000:
                ME_sfc_base = 173
            else:
                ME_sfc_base = 156
        elif engine_category == 'LNG-Otto SS':
            if built_year > 2000:
                ME_sfc_base = 148.712
        elif engine_category == 'LNG-Diesel':
            if built_year > 2000:
                ME_sfc_base = 140.3375
                
    # AE_sfc_base and Boiler_sfc_base
    AE_sfc_base = None
    Boiler_sfc_base = None
    
    if 'ifo' in fuel_type:
        if built_year < 1983:
            AE_sfc_base = 225
            Boiler_sfc_base = 340
        elif 1984 <= built_year <= 2000:
            AE_sfc_base = 205
            Boiler_sfc_base = 340
        else:
            AE_sfc_base = 195
            Boiler_sfc_base = 340
    
    elif 'mdo' in fuel_type or 'biofuel' in fuel_type:
        if built_year < 1983:
            AE_sfc_base = 210
            Boiler_sfc_base = 320
        elif 1984 <= built_year <= 2000:
            AE_sfc_base = 190
            Boiler_sfc_base = 320
        else:
            AE_sfc_base = 185
            Boiler_sfc_base = 320
    
    elif "lng" in fuel_type:
        if built_year < 1983:
            AE_sfc_base = np.nan
            Boiler_sfc_base = 285
        elif 1984 <= built_year <= 2000:
            AE_sfc_base = 173
            Boiler_sfc_base = 285
        else:
            AE_sfc_base = 156
            Boiler_sfc_base = 285

    # Return all three base values as a tuple
    return ME_sfc_base, AE_sfc_base, Boiler_sfc_base


# Load R data file and reformat MMSI
wfr_bulkers = pyreadr.read_r(os.path.join(datapath, 'bulkers_WFR.Rda'))["bulkers_df"]
wfr_bulkers = wfr_bulkers.rename(columns={'MMSI': 'mmsi'})

#%% Which speed variable to use?
# Check how many non-missing values for each candidate speed variable
speed_vars_nonNA = wfr_bulkers[['Service.Speed..knots.', 'Maximum.Speed..knots.', 'Speed..knots.']].agg(['count', 'size']).transpose()

speed_vars_nonNA['fraction'] = speed_vars_nonNA['count'] / speed_vars_nonNA['size']
print(speed_vars_nonNA)
# service speed is more complete than maximum speed
# Speed..knots. is ambiguously defined so do not use it

# IMO4 uses service speed from IHS database
# "The speed reported in the IHS dataset is called “speed.” IHS defines speed as follows: “Maximum vessel speed in knots when the ships engine is running at Maximum continuous rating (MCR).” In this report, it was assumed that on average “speed” was reporting the ship’s maximum speed at the ship maximum continuous rating (MCR)."
# "Third IMO GHG Study 2014 assumed that the reported values corresponded to 90% MCR"

# Clarkson WFR Speed variable definitions:
# Service Speed (knots): The “average” speed that a vessel maintains under normal load and weather conditions, as provided by vessel specifications or descriptions.
# Maximum Speed (knots): Maximum speed that the ship can practically achieve
# Speed (knots) is a “combined” field in which we aim to report the most relevant speed for each vessel based on the sources available. It is accompanied by “Speed Category” – indicating which type of speed has been used.

#%% Plot scatter plot of service speed vs speed
fig = sns.scatterplot(data=wfr_bulkers, x='Service.Speed..knots.', y='Speed..knots.', color = 'black').get_figure()
diagonal_coords = np.array([wfr_bulkers['Speed..knots.'].min(), wfr_bulkers['Speed..knots.'].max()])
sns.lineplot(x=diagonal_coords, y=diagonal_coords, color='grey')
fig.savefig(os.path.join(plotpath, plotprefix + 'speed_vs_servicespeed.png'))

# We choose to use 'Service.Speed..knots.' as the speed variable  due to the combination of completeness and consistent definition across all ships
#%%
# ***CLARIFY***
# derive maximum speed from service speed(service speed is 92% of maximum speed)
wfr_bulkers['Reference.Speed..knots.'] = wfr_bulkers['Service.Speed..knots.'] #/ 0.92


#%% Check how many non-missing values for other variables required for power calculation
other_vars_nonNA = wfr_bulkers[['mmsi', 'Dwt', 'Main.Engine.Detail', 'Draught..m.', 'Main.Engine.Fuel.Type', 'Built.Year']].agg(['count', 'size']).transpose()
other_vars_nonNA['fraction'] = other_vars_nonNA['count'] / other_vars_nonNA['size']
print(other_vars_nonNA)

# Only AIS identifier (MMSI) and draught are missing for some ships

#%% Drop ships that are missing variables required for power calculation (mainly MMSI is missing, which is required for matching with AIS data)
wfr_bulkers = wfr_bulkers.dropna(subset=['mmsi', 'Dwt', 'Main.Engine.Detail',
                                         'Draught..m.', 'Main.Engine.Fuel.Type',
                                         'Built.Year','Service.Speed..knots.',
                                         'Reference.Speed..knots.']) 
wfr_bulkers['mmsi'] = wfr_bulkers['mmsi'].astype('int') 

# CALCULATIONS
#%% Calculate fixed ship component of power for Admiralty formula (Equation 8 in IMO4 without t_i and v_i)
# All values as per IMO4
eta_w = np.where((wfr_bulkers['Dwt'] >= 0) & (wfr_bulkers['Dwt'] <= 9999), 0.909, 0.867) # weather correction factor
eta_f = 0.917 # fouling correction factor
m = 0.66 # draught ratio exponent
n = 3 # speed ratio exponent
delta_w = 1 # speed-power correction factor
wfr_bulkers['W_component'] = delta_w / (eta_w * eta_f * wfr_bulkers['Draught..m.']**m * wfr_bulkers['Reference.Speed..knots.']**n)

#%% Parse engine rotaional speed 'rpm' and main engine reference power 'ME_W_ref' columns from 'Main.Engine.Detail'
# and categorize engine type as per IMO4
wfr_bulkers[['Engine_Category', 'rpm', 'ME_W_ref']] = wfr_bulkers.apply(categorize_engine, axis=1)

## Check output

### Check for outliers
#%% Distribution of engine categories
print(wfr_bulkers['Engine_Category'].value_counts(dropna=False))

#%% Check for outliers in engine speeds visually
fig = sns.scatterplot(data=wfr_bulkers, x=np.log(wfr_bulkers['Dwt']), y='rpm', hue='Engine_Category').get_figure()

#%% Check for outliers in engine power visually
fig = sns.scatterplot(data=wfr_bulkers, x=np.log(wfr_bulkers['Dwt']), y='ME_W_ref', hue='Engine_Category').get_figure()
plt.figure()

fig = sns.histplot(data=wfr_bulkers, x='ME_W_ref', hue='Engine_Category', kde=True).get_figure()
plt.figure()

#%% Verify and manually fix engine power outliers (divides very high values by 10)
print(wfr_bulkers[wfr_bulkers['ME_W_ref'] > 60000][['mmsi', 'ME_W_ref', 'Main.Engine.Detail', 'Dwt']])

# Three almost identical ships have an infeasibly high power
# Check power for ships with same engine model
similar_outliers = wfr_bulkers[wfr_bulkers['Main.Engine.Detail'].str.contains('6S50MC-C8.2', na=False, case=False)][['mmsi', 'ME_W_ref', 'Main.Engine.Detail', 'Dwt']]

fig = sns.histplot(data=similar_outliers[similar_outliers['ME_W_ref'] < 60000], x='ME_W_ref').get_figure()

print(f'Mean power for engine model: {similar_outliers.ME_W_ref.mean()}')

# The outlier values are almost 10x the mean value for ships with the same engine model.
# Assume typo added a 0, so manually divide these power values by 10
wfr_bulkers.loc[wfr_bulkers['ME_W_ref'] > 60000, 'ME_W_ref'] = wfr_bulkers.loc[wfr_bulkers['ME_W_ref'] > 60000, 'ME_W_ref'] / 10

#%% Corrected plots
sns.histplot(data=wfr_bulkers[(wfr_bulkers['Dwt'] >= 38000) & (wfr_bulkers['Dwt'] <= 40000)], x='ME_W_ref', hue='Engine_Category').get_figure()
plt.figure()

sns.histplot(data=wfr_bulkers, x='ME_W_ref', hue='Engine_Category', kde=True).get_figure()
plt.figure()

sns.scatterplot(data=wfr_bulkers, x=np.log(wfr_bulkers['Dwt']), y='ME_W_ref', hue='Engine_Category', alpha=0.1).get_figure()
plt.figure()

### Check missing values
#%%
engine_vars_nonNA = wfr_bulkers[['Engine_Category', 'rpm', 'ME_W_ref']].agg(['count', 'size']).transpose()
engine_vars_nonNA['fraction'] = engine_vars_nonNA['count'] / engine_vars_nonNA['size']
engine_vars_nonNA['count_NA'] = engine_vars_nonNA['size'] - engine_vars_nonNA['count']
print(engine_vars_nonNA)

#%% RPM
missing_rpm_idx = wfr_bulkers['rpm'].isna()
wfr_bulkers.loc[missing_rpm_idx, ['IMO.Number', 'Name', 'Main.Engine.Detail']]

#%% Impute with median of entire fleet
wfr_bulkers['rpm'] = wfr_bulkers['rpm'].fillna(wfr_bulkers['rpm'].median())

#%% Plot imputed (red) and reported (blue) RPM vs log(Dwt)
fig = sns.scatterplot(data=wfr_bulkers.loc[~missing_rpm_idx], x=np.log(wfr_bulkers['Dwt']), y='rpm', color='blue', alpha=0.1).get_figure()
fig = sns.scatterplot(data=wfr_bulkers.loc[missing_rpm_idx], x=np.log(wfr_bulkers['Dwt']), y='rpm', color='red').get_figure()
# THIS IS NOT A GREAT WAY TO IMPUTE

#%% Main Engine Power
wfr_bulkers[wfr_bulkers['ME_W_ref'].isna()][['IMO.Number', 'Name', 'Main.Engine.Detail']]
# DO WE NEED TO IMPUTE?

#%% Engine Category
wfr_bulkers[wfr_bulkers['Engine_Category'].isna()][['IMO.Number', 'Name', 'Main.Engine.Detail']]
# 30 ships have missing engine category. 
# This variable is only used for SFC assignment. This is dealt with below.

# Specific fuel consumption base values
#%% Assign base SFC values for main, auxiliary, and boiler from IMO4 Table 19

# Assign for ships with non-missing 'Engine_Category'
wfr_bulkers[['ME_SFC_base', 'AE_SFC_base', 'Boiler_SFC_base']] = wfr_bulkers.apply(assign_sfc_base, axis=1, result_type='expand')

# Assign for ships with missing 'Engine_Category'

#%% For ships with diesel engines and operating on fuel oil, assign the missing Engine_Category based on the engine speed
EC_isna = wfr_bulkers['Engine_Category'].isna()
isdiesel = wfr_bulkers['Main.Engine.Detail'].str.contains('diesel', na=False, case=False)

#%%
fuel_condition = wfr_bulkers['Main.Engine.Fuel.Type'].str.contains('mdo|ifo|biofuel', na=False, case=False)
#%%
# Apply conditions to these rows
wfr_bulkers.loc[isdiesel & EC_isna & (wfr_bulkers['rpm'] <= 300) & fuel_condition, 'Engine_Category'] = 'SSD' # slow-speed diesel
wfr_bulkers.loc[isdiesel & EC_isna & (wfr_bulkers['rpm'] > 300) & (wfr_bulkers['rpm'] <= 900) & fuel_condition, 'Engine_Category'] = 'MSD' # medium-speed diesel
wfr_bulkers.loc[isdiesel & EC_isna & (wfr_bulkers['rpm'] > 900) & fuel_condition, 'Engine_Category'] = 'HSD' # high-speed diesel

#%% Check Engine_Category after diesel fuel oil engine imputation
wfr_bulkers[EC_isna][['Name', 'Main.Engine.Detail', 'Main.Engine.Fuel.Type', 'Engine_Category']]
#%%
print(wfr_bulkers['Engine_Category'].value_counts(dropna=False))
# Still 21 missing.
# NEED TO IMPUTE FOR EXTRAPOLATION?

#%% TEMPORARY (NEED TO EDIT SUBSEQUENT FILES)
wfr_bulkers = wfr_bulkers.drop(columns=['Service.Speed..knots.']).rename(columns={'Reference.Speed..knots.': 'Service.Speed..knots.', 'Engine_Category': 'Engine Category'})

#%% Save
wfr_bulkers.to_csv(os.path.join(datapath, 'bulkers_WFR_calcs.csv'), index=False)

#%%