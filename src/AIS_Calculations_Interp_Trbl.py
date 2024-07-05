# %%
group = ais_bulkers.partitions[0].loc[-7700726].compute()
group = interpolate_missing_hours(group)
group = infill_draught_partition(group)
group = pd_diff_haversine(group)
group = impute_speed(group)
group = assign_phase(group)

#%%
group = ais_bulkers.partitions[0].loc[-7700726].compute()
group = process_group(group)

#%%
part0imos = ais_bulkers.partitions[0].index.unique().compute()
part0imos

#%%
for imo in part0imos:
    print(imo)
    group = ais_bulkers.partitions[0].loc[imo].compute()
    group = process_group(group)

#%%
ais_bulkers.query('imo == -7700726').compute()

#%%
partition = ais_bulkers.partitions[0].compute()
start_time = time.time()
partition = process_partition(partition)


%%
group = ais_bulkers.partitions[0].loc[-7700726].compute()
# group = ais_bulkers.partitions[0].loc[7207530].compute()

group['interpolated'] = False

# Calculate step size for timestamp interpolation
# group['interp_steps'] = round(group.time_interval).clip(lower=1)
group['timestamp_hour'] = group.timestamp.dt.floor('H')
# group = group.loc[~group['timestamp_hour'].duplicated(keep='first')]
group['interp_steps'] = group.timestamp_hour.diff().dt.total_seconds().div(3600).fillna(1)
group['interp_step'] = (group.time_interval.clip(lower=1)/group.interp_steps).shift(-1).fillna(1)

# Calculate slerp location interpolation function
lat_rad = np.radians(group['latitude'])
long_rad = np.radians(group['longitude'])
col1 = np.cos(long_rad) * np.cos(lat_rad)
col2 = np.sin(long_rad) * np.cos(lat_rad)
col3 = np.sin(lat_rad)
rotation = R.from_rotvec(np.column_stack([col1,col2,col3]))
# if len(rotation) < 2:
#     return group
slerp = Slerp(np.arange(0, len(rotation)), rotation)

# Create row number column of type int
group['data_counter'] = np.arange(len(group))

# Create interpolated rows
group.reset_index(inplace=True)
group.set_index('timestamp_hour', inplace=True)
group = group.resample('H').asfreq()
group.imo.ffill(inplace=True)
# group.path.ffill(inplace=True)
group.reset_index(inplace=True)
group.set_index('imo', inplace=True)
group.index = group.index.astype(int)
group.timestamp.ffill(inplace=True)
group.time_interval.bfill(inplace=True)
group.interp_step.ffill(inplace=True)
group['interp_steps'] = group.interp_steps.bfill().astype(int)
group['path'] = group.path.astype(bool)
group['interpolated'] = group['interpolated'].astype(bool)
# Interpolate timestamps
group['interp_counter'] = (np.ceil((group.timestamp_hour - group.timestamp).dt.total_seconds() / 3600).astype(int))
group['timestamp'] = group.timestamp + pd.to_timedelta(group.interp_step*group.interp_counter, unit='H')

# Interpolate coordinates
group['interp_coord_index'] = group.data_counter.ffill() + group.interp_counter/group.interp_steps
slerp_vec = slerp(group.interp_coord_index).as_rotvec()
group['latitude'] = np.degrees(
    np.arctan2(
        slerp_vec[:, 2],
        np.sqrt(slerp_vec[:, 0]**2 + slerp_vec[:, 1]**2)))
group['longitude'] = np.degrees(np.arctan2(slerp_vec[:, 1], slerp_vec[:, 0]))
group.loc[group['data_counter'].isna(), 'interpolated'] = True
group = group.drop(columns=['data_counter', 'timestamp_hour', 'interp_steps', 'interp_step', 'interp_counter', 'interp_coord_index'])