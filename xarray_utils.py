import xarray as xr
import pandas as pd
import os

def analyze_netcdf(file_path: str):
    """
    Load a NetCDF file using xarray and print its important components for analysis.
    
    Args:
        file_path (str): The path to the NetCDF file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    try:
        # Load the dataset
        ds = xr.open_dataset(file_path)
        
        print("="*50)
        print(f"Analysis for NetCDF File: {file_path}")
        print("="*50)
        
        print("\n--- Dimensions ---")
        for dim, size in ds.sizes.items():
            print(f"{dim}: {size}")
            
        print("\n--- Coordinates ---")
        for coord in ds.coords:
            print(f"- {coord}:")
            print(f"    dtype: {ds.coords[coord].dtype}")
            print(f"    shape: {ds.coords[coord].shape}")
            if ds.coords[coord].attrs:
                print(f"    attributes: {ds.coords[coord].attrs}")
                
        print("\n--- Data Variables ---")
        for var in ds.data_vars:
            print(f"- {var}:")
            print(f"    dtype: {ds.data_vars[var].dtype}")
            print(f"    shape: {ds.data_vars[var].shape}")
            print(f"    dimensions: {ds.data_vars[var].dims}")
            if ds.data_vars[var].attrs:
                print(f"    attributes: {ds.data_vars[var].attrs}")
                
        print("\n--- Global Attributes ---")
        if ds.attrs:
            for attr, value in ds.attrs.items():
                print(f"{attr}: {value}")
        else:
            print("No global attributes found.")
            
        print("\n" + "="*50)
        
        # Close the dataset
        ds.close()
        
    except Exception as e:
        print(f"Error reading NetCDF file: {e}")


def zarr_to_netcdf(zarr_path: str, nc_output_path: str, **kwargs):
    """
    Convert a Zarr dataset to a NetCDF file using xarray.
    
    Args:
        zarr_path (str): Path to the input Zarr store.
        nc_output_path (str): Path where the NetCDF file will be saved.
        **kwargs: Additional keyword arguments to pass to ds.to_netcdf() (e.g., format='NETCDF4')
    """
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr store not found at {zarr_path}")
        return
        
    try:
        print(f"Loading Zarr store from: {zarr_path}")
        # Using open_dataset with engine='zarr' is recommended in modern xarray
        ds = xr.open_dataset(zarr_path, engine='zarr')
        
        print(f"Saving to NetCDF: {nc_output_path}")
        # Save to NetCDF
        ds.to_netcdf(nc_output_path, **kwargs)
        
        print("Conversion completed successfully!")
        
        # Close the dataset
        ds.close()
        
    except Exception as e:
        print(f"Error converting Zarr to NetCDF: {e}")


def find_missing_days(file_path: str, time_dim: str = 'time') -> list:
    """
    Load a dataset (NetCDF or Zarr) using xarray, extract the time dimension,
    and find any missing days between the start and end dates.
    
    Args:
        file_path (str): The path to the NetCDF file or Zarr store.
        time_dim (str): The name of the time dimension in the dataset. Default is 'time'.
        
    Returns:
        list: A list of pandas Timestamp objects representing the missing days.
    """
    if not os.path.exists(file_path):
        print(f"Error: File or directory not found at {file_path}")
        return []
        
    try:
        # Load the dataset (auto-detect if zarr)
        engine = 'zarr' if file_path.endswith('.zarr') else None
        ds = xr.open_dataset(file_path, engine=engine)
        
        if time_dim not in ds.dims and time_dim not in ds.coords:
            print(f"Error: Time dimension or coordinate '{time_dim}' not found in the dataset.")
            ds.close()
            return []
            
        # Extract the time dimension values
        times = ds[time_dim].values
        
        # Convert to a pandas DatetimeIndex and get just the unique dates (days)
        time_index = pd.DatetimeIndex(times).normalize().unique()
        
        if len(time_index) == 0:
            print("No time data found.")
            ds.close()
            return []
            
        # Create a complete date range from min to max date in the data
        start_date = time_index.min()
        end_date = time_index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D') # D for daily frequency
        
        # Find the missing days by checking the difference
        missing_days = full_date_range.difference(time_index)
        
        if len(missing_days) == 0:
            print(f"No missing days found between {start_date.date()} and {end_date.date()}.")
        else:
            print(f"Found {len(missing_days)} missing day(s) between {start_date.date()} and {end_date.date()}.")
            print("First 10 Missing days (or all if < 10):")
            for day in missing_days[:10]:
                print(f"  - {day.date()}")
            if len(missing_days) > 10:
                print(f"  ... and {len(missing_days) - 10} more.")
                
        ds.close()
        return missing_days.tolist()
        
    except Exception as e:
        print(f"Error finding missing days: {e}")
        return []

