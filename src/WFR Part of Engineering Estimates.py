"""
Calculate WFR component of engineering estimates
Input(s): bulkers_WFR.csv
Output(s): bulkers_WFR_calcs.csv
Runtime:
"""


import re
import numpy as np
import pandas as pd



# Load the WFR dataset
wfr_bulkers = pd.read_csv('/Users/oliver/Desktop/Carbon Emission Project/bulkers_WFR.csv',
                         low_memory=False)

# Drop missing values on necessary variables
wfr_bulkers = wfr_bulkers.dropna(subset=['MMSI', 'Dwt', 'Main.Engine.Detail',
                                         'Draught..m.', 'Main.Engine.Fuel.Type',
                                         'Built.Year','Service.Speed..knots.']) 

# derive maximum speed from service speed(service speed is 92% of maximum speed)
wfr_bulkers['Service.Speed..knots.'] = wfr_bulkers['Service.Speed..knots.'] / 0.92

# Rename 'MMSI' column to 'mmsi'
wfr_bulkers = wfr_bulkers.rename(columns={'MMSI': 'mmsi'})
wfr_bulkers['mmsi'] = wfr_bulkers['mmsi'].astype('int')


# In[67]:


def create_W_component(df):
    
    # assigning values to weather correction factor based on 'DWT' range
    # assumed to be 0.909 for mainly small ships and 0.867 for all other ship types and sizes
    eta_w = np.where((df['Dwt'] >= 0) & (df['Dwt'] <= 9999), 0.909, 0.867)
    
    # fouling correction factor
    eta_f = 0.917
    
    # draught ratio exponent
    m = 0.66
    
    # speed ratio exponent
    n = 3
    
    # speed-power correction factor
    # assumed to be 1 (except large container and cruise)
    delta_w = 1

    # Calculate W component and create new column
    df['W_component'] = delta_w / (eta_w * eta_f * df['Draught..m.']**m * df['Service.Speed..knots.']**n)

    return df

def categorize_engine(row):
    
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
    engine_category = row['Engine Category']
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
  

# Apply the function to create 'W_component' column
wfr_bulkers = create_W_component(wfr_bulkers)

# Apply the function to create 'Engine Category', 'rpm' and 'ME_W_ref' columns
wfr_bulkers[['Engine Category', 'rpm', 'ME_W_ref']] = wfr_bulkers.apply(categorize_engine, axis=1)

# Apply the function to create 'ME_SFC_base', 'AE_SFC_base', and 'Boiler_SFC_base' columns
wfr_bulkers[['ME_SFC_base', 'AE_SFC_base', 'Boiler_SFC_base']] = wfr_bulkers.apply(assign_sfc_base, axis=1, result_type='expand')

# Check if there are NA values
print(wfr_bulkers['Engine Category'].unique())
print(wfr_bulkers['Engine Category'].value_counts(dropna=False))

# Check which observations have NA engine category
wfr_bulkers[wfr_bulkers['Engine Category'].isna()][['Main.Engine.Detail',
                                                    'Main.Engine.Fuel.Type',
                                                    'rpm']]

# Assign median of whats observed in the data to missing values
wfr_bulkers['rpm'] = wfr_bulkers['rpm'].fillna(wfr_bulkers['rpm'].median())

# Create a mask for rows where 'Engine Category' is NaN
mask = wfr_bulkers['Engine Category'].isna()

# We confirm that all ships with missing engine category run on dissel engine
# Define fuel type condition
fuel_condition = wfr_bulkers['Main.Engine.Fuel.Type'].str.contains('mdo|ifo|biofuel', na=False, case=False)

# Apply conditions to these rows
wfr_bulkers.loc[mask & (wfr_bulkers['rpm'] <= 300) & fuel_condition, 'Engine Category'] = 'SSD'
wfr_bulkers.loc[mask & (wfr_bulkers['rpm'] > 300) & (wfr_bulkers['rpm'] <= 900) & fuel_condition, 'Engine Category'] = 'MSD'
wfr_bulkers.loc[mask & (wfr_bulkers['rpm'] > 900) & fuel_condition, 'Engine Category'] = 'HSD'

# Check again
print(wfr_bulkers['Engine Category'].unique())
print(wfr_bulkers['Engine Category'].value_counts(dropna=False))

# Save
wfr_bulkers.to_csv('/Users/oliver/Desktop/Data/bulkers_WFR_calcs.csv', index=False)
