"""
Remove waterways from EEZ and assign EU status to remaining
Input(s): EEZ_land_v2_201410.shp, Portcall_exclusion_zones.gpkg
Output(s): EEZ_EU.shp
"""


#%%
import sys, os
import pycountry
import pandas as pd
import geopandas as gpd
import numpy as np
import sys, os
from qgis.core import *

datapath = './data'

#%%
# Substract manually drawn waterways
# ----------------------------------
#%%
# Supply path to qgis install location
QgsApplication.setPrefixPath("/usr/bin/qgis", True)
# Create a reference to the QgsApplication.  Setting the
# second argument to False disables the GUI.
qgs = QgsApplication([], False)
# Load providers
qgs.initQgis()

# Include Processing module
sys.path.append('/usr/share/qgis/python/plugins') # Folder where Processing is located
from processing.core.Processing import Processing
Processing.initialize()
from processing.tools import *

#%%
EEZ_filename = 'EEZ_land_v2_201410'
outfilename = 'EEZ_exclusion'
general.run("native:difference",
    {
        'INPUT': os.path.join(datapath, EEZ_filename, EEZ_filename + '.shp'),
        'OVERLAY': os.path.join(datapath, 'Portcall_exclusion_zones.gpkg'),
        'OUTPUT': os.path.join(datapath, outfilename + '.gpkg'),
    })

#%% Remove the provider and layer registries from memory
qgs.exitQgis()



#%%
# Add EU indicator
# ----------------

# Identify countries subject to MRV (i.e. in European Economic Area)
# https://ec.europa.eu/clima/eu-action/transport-emissions/reducing-emissions-shipping-sector_en#tab-0-3
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

# This is valid until the end of 2020 (Brexit takes effect)

#%%
def country_code(key):
    try:
        x = pycountry.countries.search_fuzzy(key)
    except:
        return None
    else:  
        return x[0].alpha_3
    return x

def country_name(key):
    try:
        x = pycountry.countries.search_fuzzy(key)
    except:
        return None
    else:  
        return x[0].name
    return x

def country_correct(country):
    EU_df.loc[EU_df.name == country, 'country_code'] = pycountry.countries.search_fuzzy(country)[-1].alpha_3
    EU_df.loc[EU_df.name == country, 'name_pycountry'] = pycountry.countries.search_fuzzy(country)[-1].name

#%%
EU_df = pd.DataFrame(EU_territory, columns = ['name'])
EU_df['country_code'] = EU_df.name.apply(lambda x: country_code(x))
EU_df['name_pycountry'] = EU_df.name.apply(lambda x: country_name(x))
country_correct('Guadeloupe')
country_correct('Martinique')
if len(EU_df[EU_df.name != EU_df.name_pycountry]) > 0:
    print(EU_df[EU_df.name != EU_df.name_pycountry])
    print('Some names do not match perfectly. Verify visually!')

#%% Add EU territory indicator
EEZ_gdf = gpd.read_file(os.path.join(datapath, outfilename + '.gpkg'))

#%% Check all EU country codes are in EEZ
missing_codes = EU_df.loc[~EU_df['country_code'].isin(EEZ_gdf['ISO_3digit'])]
if len(missing_codes) != 0:
    print("The following country codes do not appear in EEZ:")
    print(missing_codes)

#%%
EEZ_gdf['EU'] = EEZ_gdf['ISO_3digit'].isin(EU_df['country_code'])

#%%
filename = 'EEZ_exclusion_EU'
outfilepath = os.path.join(datapath, filename)
if not os.path.exists(outfilepath):
    os.mkdir(outfilepath)
EEZ_gdf.to_file(
    os.path.join(outfilepath, filename + '.shp'),
    driver ='ESRI Shapefile'
    )

#%%