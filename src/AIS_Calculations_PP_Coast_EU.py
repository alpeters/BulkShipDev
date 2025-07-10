#!/usr/bin/env python3
"""
Obtain port calls by filtering potential port calls to those that are within the EU EEZ and near the coastline.

Steps:

Runtime: 2h with 6 cores
"""

import os
from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree
from dask.distributed import Client, LocalCluster
import dask_geopandas as dgpd

# Parameters
N_WORKERS = 6
THREADS_EACH = 1
LOCAL_DATAPATH = os.path.join(".", "data")
DATAPATH = os.path.join("..", "..", "SharedData")
BULK_DATAPATH = os.path.join(DATAPATH, "Bulk")
VARIANT = "speed"
BUFFER_DIST_M = 5000
PROJECTED_CRS = "EPSG:6933"
COAST_MIN_AREA = 10

# File paths
filename_base = f"potportcalls_{VARIANT}"
portcall_path = os.path.join(BULK_DATAPATH, filename_base, f"{filename_base}.shp")

eez_name = "EEZ_exclusion_EU"
eez_path = os.path.join(DATAPATH, eez_name, f"{eez_name}.shp")

coast_path = os.path.join(DATAPATH, "gshhg-shp-2.3.7", "GSHHS_shp", "f", "GSHHS_f_L1.shp")
output_csv = os.path.join(BULK_DATAPATH, f"{filename_base}_EU.csv")

# Functions
def log(msg: str) -> None:
    """Simple timestamped logger."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def spatial_join_chunk(chunk):
    return gpd.sjoin(
        chunk,
        coast[["geometry", "id"]],
        how="inner",
        predicate="intersects",
    ).drop(columns=["index_right"])

def distance_filter(points: gpd.GeoDataFrame,
                    polygons,
                    buffer_dist):
    """
    Keep only rows in *chunk* whose geometry lies within *buffer_dist* of
    any geometry in *coast_geoms*.
    """
    tree = STRtree(polygons["geometry"].values)
    pairs = tree.query(
        points.geometry.values,
        predicate="dwithin",
        distance=buffer_dist
    )
    hit_rows = np.unique(pairs[0])
    return points.iloc[hit_rows]

def compute_min_distance(points: gpd.GeoDataFrame, polygons) -> gpd.GeoDataFrame:
    """
    Compute the nearest polygon distance for each point using query_nearest,
    returning the original points with a 'min_distance' column.
    """
    # Build the spatial index for the polygons
    tree = STRtree(polygons["geometry"].values)
    
    # For each point, get the nearest polygon and its distance
    distances = {}
    for idx, pt in points.geometry.items():
        # query_nearest returns a tuple of (nearest_index, distance)
        result = tree.query_nearest(pt, return_distance=True)
        nearest_index, distance = result
        distances[idx] = distance
    
    out = points.copy()
    out["min_distance"] = pd.Series(distances)
    return out

# Main processing pipeline
if __name__ == "__main__":
    log("Starting potential port call filtering …")
    potportcalls_gdf = (
        gpd.read_file(portcall_path)
        .set_crs("EPSG:4326")  # EEZ is in 4326
    )

    eez_gdf = gpd.read_file(eez_path)[["geometry", "ISO_3digit", "EU"]]

    # Join EU status to port calls
    ## Do this join in 4326 do avoid antimeridional splitting issues in reprojecting EEZ map
    log("Assigning EEZ nationality using spatial join (within) …")
    potportcalls_eu_gdf = gpd.sjoin(
        potportcalls_gdf,
        eez_gdf,
        how="left",
        predicate="within"
    ).drop(columns=["index_right"])

    # Keep only the first match per point
    potportcalls_eu_gdf = potportcalls_eu_gdf.loc[~potportcalls_eu_gdf.index.duplicated(keep="first")]

    # Reproject to a global CRS for minimal distortion (verified no problem with antimeridional splitting)
    potportcalls_eu_gdf = potportcalls_eu_gdf.to_crs(PROJECTED_CRS)

    # Drop points outside EEZ
    len_prefilter = len(potportcalls_eu_gdf)
    potportcalls_eu_gdf = potportcalls_eu_gdf[potportcalls_eu_gdf["ISO_3digit"].notna()].copy()
    log(f"Filtered {len_prefilter - len(potportcalls_eu_gdf)} points outside EEZ.")

    # non-parallelized
    # coast_gdf = gpd.read_file(coast_path).to_crs(PROJECTED_CRS)
    # coast_gdf = coast_gdf[coast_gdf["area"] > COAST_MIN_AREA]
    # print(f"{len(coast_gdf)} coastline polygons after filtering by area > {COAST_MIN_AREA}.")
    
    # parallelized
    coast_dgdf = dgpd.read_file(coast_path, npartitions=1).to_crs(PROJECTED_CRS)
    coast_dgdf = coast_dgdf[coast_dgdf["area"] > COAST_MIN_AREA]
    print(f"{len(coast_dgdf)} coastline polygons after filtering by area > {COAST_MIN_AREA}.")

    # Spatial join to keep stops near land
    potportcalls_eu_gdf = potportcalls_eu_gdf.iloc[:1400]  # For testing
    log("Identifying stops near coastline as port calls …")
    
    # distance join, unparallelized
    # -----------
    # -----------
    # # for troubleshooting
    # potportcalls_eu_gdf = potportcalls_eu_gdf.query("imo == -9860269")
    # potportcalls_eu_gdf.to_csv('./data/potportcalls_eu_prefilter.csv', index=False)
    # compute_min_distance(
    #     potportcalls_eu_gdf,
    #     coast_gdf
    # ).to_csv('./data/min_distances.csv', index=False)
    # ----------
    # portcalls_eu_gdf = distance_filter(
    #     potportcalls_eu_gdf,
    #     coast_gdf,
    #     BUFFER_DIST_M
    # )
    
    # # distance join, parallelized with Dask
    # # -----------
    potportcalls_eu_dgdf = (
        dgpd.from_geopandas(
            potportcalls_eu_gdf,
            npartitions=N_WORKERS * THREADS_EACH
            # npartitions=2
        )
        .spatial_shuffle()
    )

    with LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_EACH
    ) as cluster, Client(cluster) as client:
        coast_dgdf_scattered = client.scatter(coast_dgdf, broadcast=True)
        potportcalls_eu_dgdf = potportcalls_eu_dgdf.persist()
        portcalls_eu_gdf = potportcalls_eu_dgdf.map_partitions(
            distance_filter,
            coast_dgdf_scattered,
            BUFFER_DIST_M,
            meta=potportcalls_eu_dgdf._meta
        ).drop(columns="geometry").compute()

    # # sjoin
    # # -----------
    # # Buffer potential port calls
    # # log(f"Buffering {len(portcalls_eu)} points by {BUFFER_DIST_M} meters …")
    # # portcalls_eu["geometry"] = portcalls_eu.geometry.buffer(
    # #     BUFFER_DIST_M,
    # #     resolution=5,
    # #     cap_style=1,
    # #     join_style=1,
    # #     mitre_limit=2,
    # # )
    # # # Reproject back to WGS84 for further processing/output
    # # portcalls_eu = portcalls_eu.to_crs("EPSG:4326")
    # # log("Saving buffered layer to GeoPackage …")
    # # portcalls_eu.to_file(buffer_gpkg, driver="GPKG")
    
    # ## manual parallelization
    # ## -----------

    # # indices = np.array_split(np.arange(len(portcalls_eu)), 1000)
    # # portcalls_eu_chunks = [portcalls_eu.iloc[idx] for idx in indices]
    
    # # with ProcessPoolExecutor() as executor:
    # #     portcalls_land_chunks = list(executor.map(spatial_join_chunk, portcalls_eu_chunks))

    # # portcalls_land = gpd.GeoDataFrame(
    # #     pd.concat(
    # #         portcalls_land_chunks, ignore_index=True
    # #     )
    # # )

    # # portcalls_land_chunks = []
    # # for chunk in portcalls_eu_chunks:
    # #     log(f"Processing chunk of size {len(chunk)} …")
    # #     chunk_land = gpd.sjoin(
    # #         chunk,
    # #         coast[["geometry", "id"]],
    # #         how="inner",
    # #         predicate="intersects",
    # #     ).drop(columns=["index_right"])

    # #     portcalls_land_chunks.append(chunk_land)
    # # portcalls_land = gpd.GeoDataFrame(pd.concat(portcalls_land_chunks, ignore_index=True))

    # ## non-parallelized
    # ## -----------
    # # portcalls_land = gpd.sjoin(
    # #     portcalls_eu,
    # #     coast[["geometry", "id"]],
    # #     how="inner",
    # #     predicate="intersects",
    # # ).drop(columns=["index_right"])

    log(f"Retained {len(portcalls_eu_gdf)} potential port calls near the coast.")
    portcalls_eu_gdf.to_csv(output_csv, index=False)
    log(f"Finished! Output saved to {output_csv}.")
