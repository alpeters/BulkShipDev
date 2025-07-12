'''
Convert raw csv data into parquet
Input(s): Spire_Cargos_AIS_01012019_31122021_hourlydownsampled_0*.csv
Output(s): ais_raw.parquet
Runtime: ~30m
'''

import os
import time
import dask.dataframe as dd
import glob

# File paths
# PARENT_PATH = os.path.join('..', '..', "SharedData")
PARENT_PATH = '/media/apeters/Extreme SSD'
input_path = os.path.join(PARENT_PATH, 'maritime_client_ubc')
output_path = os.path.join(PARENT_PATH, 'AIS', 'ais_raw')

# Functions
def convert_csv_parquet(
        files,
        outdir=os.path.join(os.getcwd(), "..", "parquetdata"),
        usecols=None,
        dtypes=None,
        datetime_col=None,
        append=True
    ):
    """Convert csv files to parquet"""
    df = dd.read_csv(
        files,
        usecols=usecols,
        dtype=dtypes,
        assume_missing=True,
        engine='c'
    )
    df[datetime_col] = df[datetime_col].str.rsplit(' U', n=1, expand=True)[0]
    df[datetime_col] = dd.to_datetime(
        df[datetime_col],
        # format = '%Y-%m-%d %H:%M:%S.%f',
        errors='coerce',
        utc=True
    )
    df.to_parquet(
        outdir,
        engine='pyarrow',
        write_index=False,
        append=append
    )

# Main processing pipeline
if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8)
    client = Client(cluster)
    print(client.dashboard_link)

    # Parsing details
    usecols = ['timestamp', 'mmsi', 'msg_type', 'latitude', 'longitude', 'speed', 'heading', 'course', 'draught', 'imo', 'name', 'length', 'collection_type']
    dtypes = {
        'timestamp': 'str',
        'mmsi' : 'int32',
        'msg_type' : 'int8',
        'latitude' : 'float32',
        'longitude' : 'float32',
        'speed' : 'float32',
        'heading' : 'float32',
        'course' : 'float32',
        'draught' : 'float32',
        'imo': 'float64',
        'name': 'str',
        'length': 'float32',
        'collection_type' : 'str'
    }
    datetime_col = 'timestamp'

    filekeystring = "Spire_Cargos_AIS_01012019_31122021_hourlydownsampled_0"
    files = glob.glob(os.path.join(input_path,'*' + filekeystring + '*'))
    # files = files[0:2]  # for testing

    # Convert
    print(f"Converting {len(files)} files from {input_path}:")
    for file in list(map(lambda x : os.path.split(x)[1], files)):
        print(file)
    start = time.time()
    convert_csv_parquet(
        files,
        output_path,
        usecols,
        dtypes,
        datetime_col = datetime_col,
        append = False
    )
    end = time.time()
    print(f"Elapsed time: {(end - start)}")

    cluster.close()
    client.close()
