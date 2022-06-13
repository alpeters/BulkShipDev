'''
Convert raw csv data into parquet
Input(s): Spire_Cargos_AIS_01012019_31122021_hourlydownsampled_0*.csv
Output(s): ais_raw.parquet
Runtime: ~30m
'''
#%%
import dask.dataframe as dd
import glob, os, time

from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=8)
client = Client(cluster)
print(client.dashboard_link)

#%%
def convert_csv_parquet(files, outdir = os.getcwd() + "parquetdata", usecols = None, dtypes = None, datetime_col = None, append = True):
    """Convert csv files to parquet"""
    df = dd.read_csv(
        files,
        usecols = usecols,
        dtype = dtypes,
        assume_missing = True,
        verbose = False,
        engine = 'c'
    )
    df[datetime_col] = df[datetime_col].str.rsplit(' U', n=1, expand=True)[0]
    df[datetime_col] = dd.to_datetime(
        df[datetime_col],
        format = '%Y-%m-%d %H:%M:%S.%f',
        utc = True)
    df.to_parquet(
        outdir,
        engine = 'fastparquet',
        write_index = False,
        append = append
    )

# Parsing details
usecols = ['timestamp', 'mmsi', 'msg_type', 'latitude', 'longitude', 'speed', 'heading', 'draught', 'imo', 'name']
dtypes = {
    'timestamp': 'str',
    'mmsi' : 'int32',
    'msg_type' : 'int8',
    'latitude' : 'float32',
    'longitude' : 'float32',
    'speed' : 'float16', # can probably reduce size using float16
    'heading' : 'float16',
    'draught' : 'float16',
    'imo': 'float64',
    'name': 'str'
}
datetime_col = 'timestamp'

# Files to convert
filepath = '/media/apeters/Extreme SSD/maritime_client_ubc'
# filepath = './src/data/AIS/ais_csv'
filekeystring = "Spire_Cargos_AIS_01012019_31122021_hourlydownsampled_0"
files = glob.glob(os.path.join(filepath,'*' + filekeystring + '*'))
# files = files[0:2]

#%%
# Convert
print(f"Converting {len(files)} files from {filepath}:")
for file in list(map(lambda x : os.path.split(x)[1], files)):
    print(file)
start = time.time()
convert_csv_parquet(files, os.path.join(os.path.split(filepath)[0], 'ais_raw'), usecols, dtypes, datetime_col = datetime_col, append = False)
end = time.time()
print(f"Elapsed time: {(end - start)}")

cluster.close()
client.close()