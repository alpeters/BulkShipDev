def assign_jump_id(df):
   
    suffix = 0  
    new_data = []
    include_current = True  
    
    for index, row in df.iterrows():
        if row['is_jump_point']:
            # Increment suffix for new segment starting after this jump point
            suffix += 1
            include_current = False  # Skip this row since it's a jump point
        else:
            include_current = True  # Include this row normally
        
        if include_current:
            new_row = row.copy()
            new_imo = f"{index}-{suffix}"
            new_row['imo'] = new_imo
            new_data.append(new_row)

    return pd.DataFrame(new_data, index=[row['imo'] for row in new_data])
