"""
Select actual portcalls as the subset of potential portcalls within a certain distance of the coast.
Assign EEZ nationality (country with EU status and with waterways excluded) to these portcalls.
Input(s): EEZ_exclusion_EU.shp, potportcalls_'callvariant'.shp
Output(s): potportcalls_'callvariant'_EU.csv (, portcalls_'callvariant'_EU_buffer.gpkg)
Runtime: 7m30

Steps:
1. Load portcalls.shp
2. load EEZ_EU.shp
3. join attributes to portcalls
4. Buffer portcalls
5. load coastline map
6. join attributes (intersection)
"""

#%%
import sys, os
from qgis.core import *

datapath = './src/data'
variant = 'speed' #'heading'

#%%
## Join EU attribute using QGIS
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
filename = 'potportcalls_' + variant
potportcalls = QgsVectorLayer(
    os.path.join(datapath, filename, filename + '.shp'),
    filename)
if not potportcalls.isValid():
    print("Layer failed to load!")

#%%
EEZ_filename = 'EEZ_exclusion_EU'
EEZ = QgsVectorLayer(
    os.path.join(datapath, EEZ_filename, EEZ_filename + '.shp'),
    EEZ_filename)
if not EEZ.isValid():
    print("Layer failed to load!")

#%% Join EU status (and country) to portcalls
join = general.run("native:joinattributesbylocation",
    {
        'INPUT': potportcalls,
        'PREDICATE':[5],
        'JOIN': EEZ,
        'JOIN_FIELDS': ['ISO_3digit', 'EU'],
        'METHOD': 1,
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'OUTPUT' : 'memory:{}'
        # 'OUTPUT': os.path.join(datapath, filename + '_EU.gpkg')
    })
potportcalls_EU = join['OUTPUT']
# 5m

#%% Filter out nulls (waterways and beyond EEZ) before buffering
potportcalls_EU.setSubsetString('"ISO_3digit" != \'null\'')

#%% Buffer around potential portcalls
buffer = general.run("native:buffer",
    {
        'INPUT': potportcalls_EU,
        'DISTANCE': 0.05,
        'SEGMENTS': 5,
        'END_CAP_STYLE': 0,
        'JOIN_STYLE': 0,
        'MITER_LIMIT': 2,
        'DISSOLVE': False,
        # 'OUTPUT' : 'memory:{}'
        'OUTPUT': os.path.join(datapath, filename + '_EU_buffer.gpkg')
    })
# potportcalls_buffer = buffer['OUTPUT']
# 38s

#%% Use filtered mapfile to speed up processing
coast_filepath = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
coast_filepath_filtered = os.path.join(datapath, coast_filepath) + '|subset="area" > 10'

#%% Join coastline to detect stops near land as portcalls, drop unmatched
join = general.run("native:joinattributesbylocation",
    {
        'INPUT': os.path.join(datapath, filename + '_EU_buffer.gpkg'),
        'PREDICATE':[0], #intersect, 4 is overlap
        'JOIN': coast_filepath_filtered,
        'JOIN_FIELDS': ['id'],
        'METHOD': 1, #first matching feature, 2 is largest overlap
        'DISCARD_NONMATCHING': True,
        'PREFIX': '',
        'OUTPUT': os.path.join(datapath, 'potportcalls_' + variant + '_EU.csv')
    })
# 1m44s

#%% Remove the provider and layer registries from memory
qgs.exitQgis()

#%%