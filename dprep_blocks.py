import os 
from uprep import tic, toc, print_timing
from uprep import tilenames_tls,riofill 
from uprep import (merge_tile_files,match_raster_to_reference,
                   clip_raster_to_template,clip_raster_to_template_extent,
                   parallel_merge_all,Fdod,raster_calc)

from glob import glob


tile12dir = '/media/ljp238/12TBWolf/BRCHIEVE/TILES12'
mosaic_dir = "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik" #mosaic
vnamelist = ['tdem_dem_egm_v_3_gap','tdem_dem_egm','edem_egm',
             'tdem_hem','esawc','esawc_x','s1', 's2', 'lgeoid','ldem','ldem_egm'
             ]

# clean this 
# compress and upload  to OndeDrive / SharePoint to use in WX2 
tdem_fs = glob("/media/ljp238/12TBWolf/ARCHIEVE/DEMsProducts/TDEMX/*/DEM/*_DEM.tif")
edem_fs  =  glob("/media/ljp238/12TBWolf/ARCHIEVE/DEMsProducts/EDEMx/TILES/comprexn/*/EDEM/*_EDEM_EGM.tif")
gdtm_v_fn = "/media/ljp238/12TBWolf/ARCHIEVE/TargetProducts/GEDI/GRID/comprexn/GEDI_L3_be/GEDI03_elev_lowestmode_mean_2019108_2022019_002_03_EPSG4326.tif"
geoid_fn = "/media/ljp238/12TBWolf/ARCHIEVE/AUXsProducts/GEOID/GLOBAL/us_nga_egm2008_1.tif"


if __name__ == '__main__':
    ti = tic()
    print(tilenames_tls)
    os.makedirs(mosaic_dir, exist_ok=True)
    print('---- Brchieve files @merge ----')
    parallel_merge_all(vnamelist, tile12dir, mosaic_dir, tilenames_tls,ncpu=10)
    # for vname in vnamelist:
    #     pattern = f"{tile12dir}/*/*{vname}.tif"
    #     dem_fs = glob(pattern)
    #     print(pattern)
    #     merge_tile_files(dem_fs,vname,mosaic_dir,tilenames_tls)
    
    print('---- Archieve files @merge ----')
    edem_egm_a_fn = merge_tile_files(edem_fs,'edem_egm_a',mosaic_dir,tilenames_tls)
    tdem_wgs_a_fn = merge_tile_files(tdem_fs,'TDEM_WGS_a',mosaic_dir,tilenames_tls)
    edem_egm_12_fn = edem_egm_a_fn.replace('.tif', '_12m.tif')

    #for vname in vnamelist:merge_tile_files(dem_fs,vname,mosaic_dir,tilenames_tls)
    if not os.path.isfile(edem_egm_12_fn):
        match_raster_to_reference(edem_egm_a_fn, tdem_wgs_a_fn, edem_egm_12_fn)
    else:
        print(f"[OK] {edem_egm_12_fn} already exists, skipping reprojecting.")

    block_geoid = f"{mosaic_dir}/EGM08.tif"
    block_geoid12 = f"{mosaic_dir}/EGM08_12m.tif"
    if os.path.isfile(block_geoid12):
        print(f"{block_geoid12} already exists, skipping creation.")
    else:
        clip_raster_to_template(geoid_fn, tdem_wgs_a_fn,block_geoid12)

    if os.path.isfile(block_geoid):
        print(f"{block_geoid} already exists, skipping creation.")
    else:
        clip_raster_to_template_extent(geoid_fn, tdem_wgs_a_fn, block_geoid)
    
    tdem_egm_a_fn = tdem_wgs_a_fn.replace("WGS", "EGM")
    if not os.path.isfile(tdem_egm_a_fn):
        print(f"{tdem_egm_a_fn} already exists, skipping creation.")
        Fdod(fine_path=tdem_wgs_a_fn, coarse_path=block_geoid12, output_path=tdem_egm_a_fn)

    block_gdtmv = f"{mosaic_dir}/TLS_GEDI03_vdtm_WGS.tif"
    block_gdtmf = f"{mosaic_dir}/TLS_GEDI03_fdtm_WGS.tif"

    if os.path.isfile(block_gdtmv):
        print(f"{block_gdtmv} already exists, skipping creation.")
    else:
        clip_raster_to_template_extent(gdtm_v_fn, tdem_wgs_a_fn, block_gdtmv)

    
    print('---- Filling files @Gdem ----')
    if not os.path.isfile(block_gdtmf):
        print(f"Filling... {block_gdtmf}")
        riofill(block_gdtmv, block_gdtmf, si=0) # replace by IWD
    
    # add egm transformation 
    print('---- Transforming Href files @Gdem ----')
    block_gdtmf_egm = f"{mosaic_dir}/TLS_GEDI03_fdtm_EGM.tif"
    if not os.path.isfile(block_gdtmf_egm):
        print(f"Transforming... {block_gdtmf_egm}")
        Fdod(fine_path=block_gdtmf, coarse_path=block_geoid, output_path=block_gdtmf_egm) 
        print(block_gdtmf_egm)

   
    # dod 
    block_dod_hr_and_lr = f"{mosaic_dir}/TLS_DOD_TDX_GLOW_EGM.tif"
    #block_dod_hr_and_lr = f"{mosaic_dir}/TLS_DOD_TDX EDEM_GLOW_EGM.tif"
    if not os.path.isfile(block_dod_hr_and_lr):
        print(f"Transforming... {block_dod_hr_and_lr}")
        # diff = fine_data - coarse_resampled
        hr_fn = "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/EDEM_EGM_A.tif"
        lr_fn = "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/TLS_GEDI03_fdtm_EGM.tif"
        #bcor_sub(fine_path=hr_fn, coarse_path=lr_fn, output_path=block_dod_hr_and_lr) 
        raster_calc(hr_fn, lr_fn, 'sub', block_dod_hr_and_lr, priority='rbpath', overwrite=False)


    tf = toc()
    print_timing(ti, tf, label="dprep_blocks.py: ")
    

