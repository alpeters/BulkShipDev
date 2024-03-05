"""
Assign EU status to map economic exclusion zones (EEZ)
We first exclude manually drawn waterway areas such as at canal entrances in order to avoid incorrectly assigning portcalls in these areas.
Input(s): EEZ_land_v2_201410.shp, Portcall_exclusion_zones.gpkg
Output(s): EEZ_exclusion.gpkg (only used for verification), EEZ_exclusion_EU (shapefile in directory)
"""


#%%
import sys, os
import pycountry
import pandas as pd
import geopandas as gpd
import numpy as np
import sys, os

datapath = './src/data'

#%% Function definitions
def country_code(key):
    '''
    Return 3 letter country code corresponding to a country name using fuzzy search with pycountry function
    
    Args:
        String: country name
    Returns:
        String: 3 letter country code
    '''
    try:
        x = pycountry.countries.search_fuzzy(key)
    except:
        return None
    else:  
        return x[0].alpha_3

def country_name(key):
    '''
    Return country name corresponding to 3 letter country code using fuzzy search with pycountry function
    
    Args:
        String: 3 letter country code
    Returns:
        String: country name
    '''
    try:
        x = pycountry.countries.search_fuzzy(key)
    except:
        return None
    else:  
        return x[0].name

def country_correct(country):
    '''
    Replace country_code and name_pycountry columns of EU_df with last returned match from pycountry fuzzy search
    
    Args:
        String: country name in EU_df
    Returns:
        None (directly modifies EU_df)
    '''
    EU_df.loc[EU_df.name == country, 'country_code'] = pycountry.countries.search_fuzzy(country)[-1].alpha_3
    EU_df.loc[EU_df.name == country, 'name_pycountry'] = pycountry.countries.search_fuzzy(country)[-1].name

#%%
# Remove waterways from EEZ map
# These are primarily entrances to canals where 
# ships often stop to wait but these are not portcalls
# Portcall_exclusion_zones.gpkg contains these manually drawn areas
# ----------------------------------
EEZ_filename = 'EEZ_land_v2_201410'
outfilename = 'EEZ_exclusion'
EEZ_gdf = gpd.read_file(os.path.join(datapath, EEZ_filename, EEZ_filename + '.shp'))
exclusion_zones_gdf = gpd.read_file(os.path.join(datapath, 'Portcall_exclusion_zones.gpkg'))
difference_gdf = gpd.overlay(EEZ_gdf, exclusion_zones_gdf, how='difference')
difference_gdf.to_file(os.path.join(datapath, outfilename + '.gpkg'), driver='GPKG')

#%%
# Add EU indicator to EEZ map
# ---------------------------

# Identify countries subject to MRV (i.e. in European Economic Area) before Brexit (end of 2020)
# https://ec.europa.eu/clima/eu-action/transport-emissions/reducing-emissions-shipping-sector_en#tab-0-3
# This includes all territories that were ever included in the MRV regulation
#%%
EU_territory = [
    'Belgium',
    'Bulgaria',
    'Croatia',
    'Republic of Cyprus',
    'Denmark',
    'Estonia',
    'Finland',
    'France',
    'Germany',
    'Greece',
    'Ireland',
    'Italy',
    'Latvia',
    'Lithuania',
    'Malta',
    'Netherlands',
    'Poland',
    'Portugal',
    'Romania',
    'Slovenia',
    'Spain',
    'Sweden',
    'United Kingdom',
    'Iceland',
    'Norway',
    'Açores',
    'Canarias',
    'French Guiana',
    'Guadeloupe',
    'Madeira',
    'Martinique',
    'Mayotte',
    'Reunion',
    'Saint Martin'
]

#%%
EU_df = pd.DataFrame(EU_territory, columns = ['name'])
EU_df['country_code'] = EU_df.name.apply(lambda x: country_code(x))
#%%
EU_df['name_pycountry'] = EU_df.name.apply(lambda x: country_name(x))
#%% Replace code and name for two countries with more precise ones
country_correct('Guadeloupe')
country_correct('Martinique')

#%% Check if all pycountry reverse lookup names match original names from EU list to ensure proper country assignment
if len(EU_df[EU_df.name != EU_df.name_pycountry]) > 0:
    print(EU_df[EU_df.name != EU_df.name_pycountry])
    print('Some names do not match perfectly. Verify visually!')
# Only differences should be minor name differences or assignment of some territories to their parent countries, e.g. Canarias to Spain

#%% Add EU territory indicator
EEZ_gdf = gpd.read_file(os.path.join(datapath, outfilename + '.gpkg'))

#%% Check that all EU country codes are in EEZ map
missing_codes = EU_df.loc[~EU_df['country_code'].isin(EEZ_gdf['ISO_3digit'])]
if len(missing_codes) != 0:
    print("The following country codes do not appear in EEZ:")
    print(missing_codes)
else:
    print("All EU country codes were found in the EEZ map")

#%% Pre-Brexit includes UK
EEZ_gdf['EU_preBrexit'] = EEZ_gdf['ISO_3digit'].isin(EU_df['country_code'])
#%% Post-Brexit excludes UK
EU_postBrexit_df = EU_df[EU_df.name != 'United Kingdom']
EEZ_gdf['EU_postBrexit'] = EEZ_gdf['ISO_3digit'].isin(EU_postBrexit_df['country_code'])

#%% TEMPORARY to ensure compatibility until subsequent code is updated
EEZ_gdf['EU'] = EEZ_gdf['EU_preBrexit']

#%% Save to shapefile
filename = 'EEZ_exclusion_EU'
outfilepath = os.path.join(datapath, filename)
if not os.path.exists(outfilepath):
    os.mkdir(outfilepath)
EEZ_gdf.to_file(
    os.path.join(outfilepath, filename + '.shp'),
    driver ='ESRI Shapefile'
    )

#%%