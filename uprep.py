import os 
import sys 
import time
import numpy as np
import warnings
import rasterio
from glob import glob
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from concurrent.futures import ProcessPoolExecutor

from uvars import gtdx_dir
sys.path.append(gtdx_dir)
from ufuncs import mosaic
from utilenames import tilenames_tls
from uinterp import riofill

warnings.filterwarnings("ignore", category=UserWarning, module='rasterio')

def tic():
    """
    Start a timer and return the current time in seconds since the epoch.
    """
    return time.time()

def toc(): 
    return time.time()

def print_timing(start_time, end_time, label=""):
    elapsed = end_time - start_time
    start_str = f"{label}Start time: {time.ctime(start_time)}"
    end_str = f"{label}End time: {time.ctime(end_time)}"
    elapsed_str = (f"{label}Elapsed time: {elapsed:.2f} sec | "
                   f"{elapsed/60:.2f} min | "
                   f"{elapsed/3600:.2f} hrs | "
                   f"{elapsed/86400:.2f} days")
    # return them as a tuple
    print(start_str)
    print(end_str)
    print(elapsed_str)
    return start_str, end_str, elapsed_str



# def print_timing(start_time, end_time, label=""):
#     elapsed = end_time - start_time
#     print(f"{label}Start time: {time.ctime(start_time)}")
#     print(f"{label}End time: {time.ctime(end_time)}")
#     print(f"{label}Elapsed time: {elapsed:.2f} sec | {elapsed/60:.2f} min | {elapsed/3600:.2f} hrs | {elapsed/86400:.2f} days")
#     return 

def task_merge_tile(vname, tile12dir, mosaic_dir, tilenames_tls):
    pattern = f"{tile12dir}/*/*{vname}.tif"
    dem_fs = glob(pattern)
    print(f"Processing pattern: {pattern}, found {len(dem_fs)} files.")
    merge_tile_files(dem_fs, vname, mosaic_dir, tilenames_tls)

def parallel_merge_all(vnamelist, tile12dir, mosaic_dir, tilenames_tls, ncpu=10):
    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        futures = [
            executor.submit(task_merge_tile, vname, tile12dir, mosaic_dir, tilenames_tls)
            for vname in vnamelist
        ]
        for f in futures:
            f.result()


def get_tile_files(dem_fs,vname,tilenames=tilenames_tls):
    dem_ts = [i for i in dem_fs for j in tilenames if j in i]
    vtfile= dem_ts
    print(f'length of vtfile: {len(vtfile)}')
    assert len(vtfile) == len(tilenames), f"Error: {vname} not found in tfiles"
    return vtfile

def merge_tile_files(tile12dir:str,vname:str,mosaic_dir:str,tilenames:list=tilenames_tls):
    print(f"Processing {vname} files...")
    vtfile = get_tile_files(tile12dir,vname,tilenames)
    if not vtfile:
        print(f"No files found for {vname}. Skipping...")
        sys.exit(0)
    print('----Mosaic the files----')
    vblock_tif = f"{mosaic_dir}/{vname.upper()}.tif"
    print(f"vname: {vblock_tif}")
    if not os.path.isfile(vblock_tif):
        mosaic(input_files=vtfile, output_file=vblock_tif)
    return vblock_tif


def match_raster_to_reference(source_path, reference_path, output_path):
    with rasterio.open(reference_path) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_width = ref_ds.width
        ref_height = ref_ds.height
        ref_res = (ref_transform.a, -ref_transform.e)

    with rasterio.open(source_path) as src_ds:
        needs_update = False
        src_crs = src_ds.crs
        src_transform = src_ds.transform
        src_width = src_ds.width
        src_height = src_ds.height
        src_res = (src_transform.a, -src_transform.e)

        # Check CRS, shape, resolution
        if (src_crs != ref_crs or
            src_width != ref_width or
            src_height != ref_height or
            src_res != ref_res):
            needs_update = True

        if needs_update:
            print(f"[INFO] Reprojecting {os.path.basename(source_path)} to match {os.path.basename(reference_path)}")

            kwargs = src_ds.meta.copy()
            kwargs.update({
                'crs': ref_crs,
                'transform': ref_transform,
                'width': ref_width,
                'height': ref_height
            })

            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src_ds.count + 1):
                    reproject(
                        source=rasterio.band(src_ds, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
            print(f"[DONE] Saved aligned raster to: {output_path}")
        else:
            print("[OK] Files already match — no new file created.")


def clip_raster_to_template(big_raster_path, template_path, output_path):
    with rasterio.open(template_path) as template_ds:
        template_bounds = template_ds.bounds
        template_transform = template_ds.transform
        template_crs = template_ds.crs
        template_shape = (template_ds.height, template_ds.width)

        with rasterio.open(big_raster_path) as big_ds:
            # Reproject big raster to match the template if needed
            if big_ds.crs != template_crs:
                transform, width, height = calculate_default_transform(
                    big_ds.crs, template_crs,
                    big_ds.width, big_ds.height,
                    *big_ds.bounds)
                
                kwargs = big_ds.meta.copy()
                kwargs.update({
                    'crs': template_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                reprojected_path = 'temp_reprojected.tif'
                with rasterio.open(reprojected_path, 'w', **kwargs) as dst:
                    for i in range(1, big_ds.count + 1):
                        reproject(
                            source=rasterio.band(big_ds, i),
                            destination=rasterio.band(dst, i),
                            src_transform=big_ds.transform,
                            src_crs=big_ds.crs,
                            dst_transform=transform,
                            dst_crs=template_crs,
                            resampling=Resampling.bilinear)

                # Open the reprojected raster for clipping
                src = rasterio.open(reprojected_path)
            else:
                src = big_ds

            # Get window from template bounds
            window = from_bounds(*template_bounds, transform=src.transform)
            window = window.round_offsets().round_shape()

            # Read data from that window
            data = src.read(window=window, out_shape=(src.count, *template_shape), resampling=Resampling.bilinear)

            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "height": template_shape[0],
                "width": template_shape[1],
                "transform": template_transform,
                "crs": template_crs
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(data)

            # Clean up temp file if used
            if src.name != big_ds.name:
                src.close()
                os.remove(reprojected_path)


def clip_raster_to_template_extent(big_raster_path, template_path, output_path):
    try:
        with rasterio.open(template_path) as template_ds:
            template_bounds = template_ds.bounds
            template_crs = template_ds.crs

        with rasterio.open(big_raster_path) as big_ds:
            big_crs = big_ds.crs

            # Reproject bounds if CRS mismatch
            if big_crs != template_crs:
                print("[INFO] Reprojecting template bounds to match big raster CRS.")
                template_bounds = transform_bounds(template_crs, big_crs, *template_bounds)

            # Clip window from reprojected bounds
            window = from_bounds(*template_bounds, transform=big_ds.transform).round_offsets()

            # Read clipped data
            data = big_ds.read(window=window)
            out_transform = big_ds.window_transform(window)

            # Update metadata (keep everything else the same)
            out_meta = big_ds.meta.copy()
            out_meta.update({
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": out_transform
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(data)

            print(f"[DONE] Clipped raster saved to: {output_path}")

    except RasterioIOError as e:
        print(f"[ERROR] File issue: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")


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

def Fdod(fine_path, coarse_path, output_path):

    if os.path.isfile(output_path):
        print(f"[OK] Output file '{output_path}' already exists. Skipping operation.")
        return
    coarse_resampled, profile = resample_raster(coarse_path, fine_path)

    with rasterio.open(fine_path) as fine_ds:
        fine_data = fine_ds.read().astype(np.float32)

    # Subtract (fine - coarse)
    diff = fine_data - coarse_resampled

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff)

def raster_calc(
    raster1_path: str,
    raster2_path: str,
    operation: str,
    output_path: str,
    priority: str = 'rapath', # 'rapath' for raster1 priority, 'rbpath' for raster2 priority
    overwrite: bool = False
) -> None:
    """
    Performs an arithmetic operation (add, sub, mul, div) on two raster files.

    The rasters can be reprojected and aligned to match the CRS, bounds, and resolution
    of a 'priority' raster.

    Args:
        raster1_path (str): Path to the first input raster file.
        raster2_path (str): Path to the second input raster file.
        operation (str): The arithmetic operation to perform ('add', 'sub', 'mul', 'div').
        output_path (str): Path where the output raster will be saved.
        priority (str): Determines which raster's properties (CRS, transform, resolution)
                        will be used for alignment.
                        'rapath' (default): raster1_path properties are prioritized.
                        'rbpath': raster2_path properties are prioritized.
        overwrite (bool): If True, overwrites the output_path if it already exists.
                          If False and output_path exists, the function will print a message and exit.
    """
    # 1. Check if output_path exists and handle overwrite option
    if os.path.exists(output_path):
        if overwrite:
            print(f"Output file '{output_path}' exists. Overwriting...")
            os.remove(output_path)
        else:
            print(f"Output file '{output_path}' already exists. Skipping operation. "
                  "Set 'overwrite=True' to force overwrite.")
            return

    # Validate operation
    valid_operations = ['add', 'sub', 'mul', 'div']
    if operation not in valid_operations:
        print(f"Error: Invalid operation '{operation}'. "
              f"Valid operations are: {', '.join(valid_operations)}")
        return

    try:
        # 2. Open rasters
        with rasterio.open(raster1_path) as src1, \
             rasterio.open(raster2_path) as src2:

            # Determine the target profile (CRS, transform, resolution)
            # This profile will define the output raster's spatial properties.
            if priority == 'rbpath':
                target_profile_src = src2
                priority_name = "raster2"
            elif priority == 'rapath':
                target_profile_src = src1
                priority_name = "raster1"
            else:
                print(f"Error: Invalid priority '{priority}'. "
                      "Must be 'rapath' or 'rbpath'. Using 'rapath' as default.")
                target_profile_src = src1
                priority_name = "raster1"

            print(f"Prioritizing properties from {priority_name} for reprojection and alignment.")

            # Prepare destination arrays for reprojected data
            # Both input rasters will be reprojected to the shape and properties of target_profile_src
            height, width = target_profile_src.shape
            data_r1_aligned = np.empty((height, width), dtype=np.float32)
            data_r2_aligned = np.empty((height, width), dtype=np.float32)

            # Reproject src1 to the target profile
            print(f"Reprojecting '{src1.name}' to match '{target_profile_src.name}'...")
            reproject(
                source=rasterio.band(src1, 1),        # Source raster band
                destination=data_r1_aligned,          # Destination numpy array for raster1 data
                src_transform=src1.transform,         # Source transform of raster1
                src_crs=src1.crs,                     # Source CRS of raster1
                dst_transform=target_profile_src.transform, # Destination transform (from priority)
                dst_crs=target_profile_src.crs,       # Destination CRS (from priority)
                resampling=Resampling.nearest,        # Resampling method
                num_threads=2                         # Number of threads for parallel processing
            )
            print("Reprojection of raster1 complete.")

            # Reproject src2 to the target profile
            print(f"Reprojecting '{src2.name}' to match '{target_profile_src.name}'...")
            reproject(
                source=rasterio.band(src2, 1),        # Source raster band
                destination=data_r2_aligned,          # Destination numpy array for raster2 data
                src_transform=src2.transform,         # Source transform of raster2
                src_crs=src2.crs,                     # Source CRS of raster2
                dst_transform=target_profile_src.transform, # Destination transform (from priority)
                dst_crs=target_profile_src.crs,       # Destination CRS (from priority)
                resampling=Resampling.nearest,        # Resampling method
                num_threads=2                         # Number of threads for parallel processing
            )
            print("Reprojection of raster2 complete.")

            # Now, data_r1_aligned and data_r2_aligned are both aligned to the chosen priority's spatial properties.

            # 3. Perform the operation
            print(f"Performing '{operation}' operation...")
            result_data = None
            if operation == 'add':
                # Result is raster1_aligned + raster2_aligned
                result_data = data_r1_aligned + data_r2_aligned
            elif operation == 'sub':
                # Always raster1_aligned - raster2_aligned as requested
                result_data = data_r1_aligned - data_r2_aligned
            elif operation == 'mul':
                # Result is raster1_aligned * raster2_aligned
                result_data = data_r1_aligned * data_r2_aligned
            elif operation == 'div':
                # Result is raster1_aligned / raster2_aligned. Handle division by zero.
                result_data = np.divide(data_r1_aligned, data_r2_aligned,
                                        out=np.full_like(data_r1_aligned, np.nan, dtype=np.float32),
                                        where=data_r2_aligned!=0)
            else:
                # This should ideally not be reached due to earlier validation
                print("Error: Unknown operation.")
                return

            print("Operation complete. Writing results...")

            # 4. Write results to opath
            # Update the profile for the output raster based on the target_profile_src
            out_profile = target_profile_src.profile
            out_profile.update(
                dtype=result_data.dtype,
                height=result_data.shape[0],
                width=result_data.shape[1],
                count=1 # Assuming single band output
            )

            with rasterio.open(output_path, 'w', **out_profile) as dst:
                dst.write(result_data, 1) # Write the result to the first band

            print(f"Operation successfully completed. Result saved to: {output_path}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading or writing raster file: {e}")
    except ValueError as e:
        print(f"Data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
