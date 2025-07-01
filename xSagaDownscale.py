import os
import time
from datetime import datetime
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np


def print_context(text):
    print('-'*40)
    print(f'{text.upper()}....')
    print('')


def gwrdownxcale(xpath, ypath, opath, geoid_fn, 
                 overwrite=False, oaux=False, epsg_code=4979, clean=True,
                 search_range=0, search_radius=10, 
                 dw_weighting=0, dw_idw_power=2.0, dw_bandwidth=1.0,
                 logistic=0, model_out=0, grid_system=None,
                 fmin_run=True):
    
    ti = time.perf_counter()
    start_time = datetime.now()

    print_context("1. setting output paths")
    opath_base, ext = os.path.splitext(opath)
    extname = ext.replace('.tif', '') if ext == '.tif' else ext.replace('.sdat', '')
    gwrp_fn_base = f"{opath_base}_dw{dw_weighting}{extname}"
    gwrp_fn = f"{gwrp_fn_base}.tif"
    print(gwrp_fn)
    print_context("2.  gwr modelling....")
    if not os.path.isfile(gwrp_fn):
        print_context("2.1. running gwr downscaling....")
        gwr_grid_downscaling(xpath, ypath, opath=gwrp_fn, oaux=oaux, 
                             epsg_code=epsg_code, clean=clean,
                            search_range=search_range, search_radius=search_radius, 
                            dw_weighting=dw_weighting, dw_idw_power=dw_idw_power, dw_bandwidth=dw_bandwidth, 
                            logistic=logistic,model_out=model_out, grid_system=grid_system)
    else:
        print_context("2.1 already exists gwr downscaled....")

    fmin_fn = f"{gwrp_fn_base}_fmin.tif"
    if fmin_run:
        if not os.path.isfile(fmin_fn):
            print_context("3.1. running fmin postprocessing....")
            fmin_get(xpath, gwrp_fn, fmin_fn) # xpath must be tdx
        else:
            print_context("3.1 already exists fmin postprocessing....")

    # print('bcor_fn...')
    # bcor_sub(gwrp_fn, geoid_fn, bcor_fn)

    # print('fminbcor_fn...')
    # fmin_get(xpath, bcor_fn, fminbcor_fn) # xpath must be tdx
    
    tf = time.perf_counter() - ti
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"stime:{start_time}\netime:{end_time}")
    print(f"runtime:{tf/60}\nduration:{duration}")

    return gwrp_fn, fmin_fn

# def gwrdownxcale(xpath, ypath, opath, geoid_fn, overwrite=False, oaux=False, epsg_code=4979, clean=True,
#                  search_range=0, search_radius=10, dw_weighting=0, dw_idw_power=2.0, dw_bandwidth=1.0,
#                  logistic=0, model_out=0, grid_system=None):
#     """
#     Performs Geographically Weighted Regression (GWR) for grid downscaling.

#     Parameters:
#     - xpath (str): Path to the high-resolution DEM (predictor variable).
#     - ypath (str): Path to the coarse-resolution data (dependent variable).
#     - opath (str): Path to save the base output SAGA grid (.sdat file).
#     - geoid_fn (str): Path to the geoid file for orthometric height conversion.
#     - overwrite (bool, optional): If True, re-runs processing and overwrites existing files.
#                                      If False, returns the expected output paths without running.
#                                      Defaults to False.
#     - oaux (bool, optional): If True, generate additional outputs like regression correction, quality, and residuals. Defaults to False.
#     - epsg_code (int, optional): EPSG code for the spatial reference system of the output GeoTIFF. Defaults to 4979.
#     - clean (bool, optional): If True, remove intermediate SAGA files after conversion. Defaults to True.
#     - search_range (int, optional): Defines the search range for GWR.
#       - 0: local
#       - 1: global
#       Defaults to 0 (local).
#     - search_radius (int, optional): Search distance in cells for local GWR. Minimum: 1. Defaults to 10.
#     - dw_weighting (int, optional): Defines the distance weighting function.
#       - 0: no distance weighting
#       - 1: inverse distance to a power
#       - 2: exponential
#       - 3: gaussian
#       Defaults to 0 (no distance weighting).
#     - dw_idw_power (float, optional): Power parameter for inverse distance weighting. Minimum: 0.0. Defaults to 2.0.
#     - dw_bandwidth (float, optional): Bandwidth for exponential and Gaussian weighting. Minimum: 0.0. Defaults to 1.0.
#     - logistic (int, optional): Enable logistic regression (Boolean: 0 for False, 1 for True). Defaults to 0.
#     - model_out (int, optional): Output the model parameters (Boolean: 0 for False, 1 for True). Defaults to 0.
#     - grid_system (str, optional): Path to a SAGA grid system file to be used. If None, the grid system is determined from the input data. Defaults to None.

#     Returns:
#     - list: A list containing the paths to the output GeoTIFF files:
#             [downscaled_dem.tif, filled_min_dem.tif, bias_corrected_dem.tif, filled_min_bias_corrected_dem.tif]
#     """
#     opath_base, ext = os.path.splitext(opath)
#     gwrp_fn_base = f"{opath_base}_dw{dw_weighting}{ext.replace('.sdat', '')}"
#     gwrp_fn = f"{gwrp_fn_base}.tif"
#     fmin_fn = f"{gwrp_fn_base}_fmin.tif"
#     bcor_fn = f"{gwrp_fn_base}_bcor.tif"
#     fminbcor_fn = f"{gwrp_fn_base}_fminbcor.tif"
#     outpaths = [gwrp_fn, fmin_fn, bcor_fn, fminbcor_fn]

#     if not overwrite:
#         return outpaths

#     ti = time.perf_counter()
#     start_time = datetime.now()

#     print('gwr_grid_downscaling...')
#     gwr_grid_downscaling_output = gwr_grid_downscaling(xpath, ypath, opath, oaux=oaux, epsg_code=epsg_code, clean=clean,
#                                      search_range=search_range, search_radius=search_radius, dw_weighting=dw_weighting,
#                                      dw_idw_power=dw_idw_power, dw_bandwidth=dw_bandwidth, logistic=logistic,
#                                      model_out=model_out, grid_system=grid_system)
#     gwrp_fn = gwr_grid_downscaling_output

#     print('fmin_fn...')
#     fmin_get(xpath, gwrp_fn, fmin_fn) # xpath must be tdx

#     print('bcor_fn...')
#     bcor_sub(gwrp_fn, geoid_fn, bcor_fn)

#     print('fminbcor_fn...')
#     fmin_get(xpath, bcor_fn, fminbcor_fn) # xpath must be tdx

#     tf = time.perf_counter() - ti
#     end_time = datetime.now()
#     duration = end_time - start_time
#     minutes = int(duration.total_seconds() // 60)
#     seconds = int(duration.total_seconds() % 60)
#     hours = int(minutes // 60)
#     remaining_minutes = int(minutes % 60)
#     days = int(hours // 24)
#     remaining_hours = int(hours % 24)

#     time_taken_str = f'{seconds} seconds'
#     if remaining_minutes > 0:
#         time_taken_str = f'{remaining_minutes} minutes, ' + time_taken_str
#     if remaining_hours > 0:
#         time_taken_str = f'{remaining_hours} hours, ' + time_taken_str
#     if days > 0:
#         time_taken_str = f'{days} days, ' + time_taken_str

#     print(f'Start time: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
#     print(f'End time: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
#     print(f'RUN.TIME = {time_taken_str}')

#     return outpaths


def gwr_grid_downscaling(xpath, ypath, opath, oaux=False, epsg_code=4979, clean=True,
                         search_range=0, search_radius=10, dw_weighting=0, dw_idw_power=2.0, dw_bandwidth=1.0,
                         logistic=0, model_out=0, grid_system=None):
    """
    Perform Geographically Weighted Regression (GWR) for grid downscaling.

    Parameters:
    - xpath (str): Path to the high-resolution DEM (predictor variable).
    - ypath (str): Path to the coarse-resolution data (dependent variable).
    - opath (str): Path to save the output SAGA grid (.sdat file).
    - oaux (bool, optional): If True, generate additional outputs like regression correction, quality, and residuals. Defaults to False.
    - epsg_code (int, optional): EPSG code for the spatial reference system of the output GeoTIFF. Defaults to 4979.
    - clean (bool, optional): If True, remove intermediate SAGA files after conversion. Defaults to True.
    - search_range (int, optional): Defines the search range for GWR.
      - 0: local
      - 1: global
      Defaults to 0 (local).
    - search_radius (int, optional): Search distance in cells for local GWR. Minimum: 1. Defaults to 10.
    - dw_weighting (int, optional): Defines the distance weighting function.
      - 0: no distance weighting
      - 1: inverse distance to a power
      - 2: exponential
      - 3: gaussian
      Defaults to 0 (no distance weighting).
    - dw_idw_power (float, optional): Power parameter for inverse distance weighting. Minimum: 0.0. Defaults to 2.0.
    - dw_bandwidth (float, optional): Bandwidth for exponential and Gaussian weighting. Minimum: 0.0. Defaults to 1.0.
    - logistic (int, optional): Enable logistic regression (Boolean: 0 for False, 1 for True). Defaults to 0.
    - model_out (int, optional): Output the model parameters (Boolean: 0 for False, 1 for True). Defaults to 0.
    - grid_system (str, optional): Path to a SAGA grid system file to be used. If None, the grid system is determined from the input data. Defaults to None.

    Returns:
    - str: Path to the output GeoTIFF file.
    """

    opath_base, ext = os.path.splitext(opath)
    #otif = f"{opath_base}_dw{dw_weighting}{ext.replace('.sdat', '.tif')}"

    # Construct the base SAGA command
    cmd = (
        f"saga_cmd statistics_regression 14 "
        f"-PREDICTORS {xpath} "
        f"-DEPENDENT {ypath} "
        f"-REGRESSION {opath} "
        f"-SEARCH_RANGE {search_range} "
        f"-SEARCH_RADIUS {search_radius} "
        f"-DW_WEIGHTING {dw_weighting} "
        f"-DW_IDW_POWER {dw_idw_power} "
        f"-DW_BANDWIDTH {dw_bandwidth} "
        f"-LOGISTIC {logistic} "
        f"-MODEL_OUT {model_out}"
    )

    if grid_system:
        cmd += f" -GRID_SYSTEM {grid_system}"

    if oaux:
        # Add optional outputs for residual correction, quality, and residuals
        opath_rescorr = f"{opath_base}_RESCORR_dw{dw_weighting}.sdat"
        opath_quality = f"{opath_base}_QUALITY_dw{dw_weighting}.sdat"
        opath_residuals = f"{opath_base}_RESIDUALS_dw{dw_weighting}.sdat"
        cmd += (
            f" -REG_RESCORR {opath_rescorr} "
            f"-QUALITY {opath_quality} "
            f"-RESIDUALS {opath_residuals}"
        )

    # Run the SAGA command
    os.system(cmd)

    # Convert the output SAGA grid to GeoTIFF
    #sdat_to_geotif(opath, otif, epsg_code)

    print("GWR Grid Downscaling completed.")
    if oaux:
        print(f"Additional outputs saved: \n{opath_rescorr.replace('.sdat', '.tif')}, \n{opath_quality.replace('.sdat', '.tif')}, \n{opath_residuals.replace('.sdat', '.tif')}")

    if clean:
        time.sleep(1)

        dirpath = os.path.dirname(opath)
        print(f'Cleaning up intermediate files...\n{dirpath}')
        for f in os.listdir(dirpath):
            if not f.endswith('.tif'):
                fo = os.path.join(dirpath, f)
                if os.path.isfile(fo):  # Check if it's a file
                    print(f'Removing {fo}...')
                    os.remove(fo)
                else:
                    print(f'Skipping directory: {fo}')
    #return otif


def sdat_to_geotif(sdat_path, gtif_path, epsg_code=4979):
    """
    Converts a Saga .sdat file to a GeoTIFF file using GDAL.

    Parameters:
        sdat_path (str): Path to the input .sdat file.
        gtif_path (str): Path to the output GeoTIFF file.
        epsg_code (int): EPSG code for the spatial reference system. Default is 4979.
    """
    # Ensure the input file has the correct extension
    if not sdat_path.endswith('.sdat'):
        sdat_path = sdat_path.replace('.sgrd', '.sdat')

    # Check if the output file already exists
    if os.path.isfile(gtif_path):
        print(f'! The file "{gtif_path}" already exists.')
        return

    # Construct and execute the GDAL command
    cmd = f'gdal_translate -a_srs EPSG:{epsg_code} -of GTiff "{sdat_path}" "{gtif_path}"'
    result = os.system(cmd)

    if result == 0:
        print(f'# Successfully converted "{sdat_path}" to "{gtif_path}".')
    else:
        print(f'! Failed to convert "{sdat_path}" to "{gtif_path}". Check the input files and GDAL installation.')


def resample_raster(src_path, match_path, resampling=Resampling.bilinear):
    with rasterio.open(match_path) as match_ds:
        match_transform = match_ds.transform
        match_crs = match_ds.crs
        match_width = match_ds.width
        match_height = match_ds.height

        with rasterio.open(src_path) as src_ds:
            data = np.empty((src_ds.count, match_height, match_width), dtype=np.float32)

            for i in range(src_ds.count):
                reproject(
                    source=rasterio.band(src_ds, i + 1),
                    destination=data[i],
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    dst_transform=match_transform,
                    dst_crs=match_crs,
                    resampling=resampling
                )

            profile = match_ds.profile.copy()
            profile.update({
                'height': match_height,
                'width': match_width,
                'transform': match_transform,
                'dtype': 'float32'
            })

            return data, profile

def bcor_sub(fine_path, coarse_path, output_path):
    coarse_resampled, profile = resample_raster(coarse_path, fine_path)

    with rasterio.open(fine_path) as fine_ds:
        fine_data = fine_ds.read().astype(np.float32)

    # Subtract (fine - coarse)
    diff = fine_data - coarse_resampled

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff)

def fmin_get(raster_a_path, raster_b_path, output_path):
    """
    Create a new raster where each pixel is the minimum of the corresponding pixels in two input rasters.
    Nodata values are treated as np.nan.

    Parameters:
        raster_a_path (str): File path to the first input raster.
        raster_b_path (str): File path to the second input raster.
        output_path (str): File path to save the output raster.
    """
    if os.path.isfile(output_path):
        print(f'! The file "{output_path}" already exists.')
        return output_path
    # Open raster a
    with rasterio.open(raster_a_path) as src_a:
        a = src_a.read(1).astype(float)
        a[a == src_a.nodata] = np.nan
        profile = src_a.profile.copy()

    # Open raster b
    with rasterio.open(raster_b_path) as src_b:
        b = src_b.read(1).astype(float)
        b[b == src_b.nodata] = np.nan

    # Compute pixel-wise minimum, treating np.nan properly
    c = np.fmin(a, b)

    # Set a float nodata value if needed (optional)
    nodata_value = -9999.0
    c[np.isnan(c)] = nodata_value
    profile.update(dtype='float32', nodata=nodata_value)

    # Write output raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(c.astype('float32'), 1)