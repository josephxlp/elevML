import os
import time 
from datetime import datetime
from glob import glob
from xSagaDownscale import gwrdownxcale
from xEnsembles import ensemble_prediction

def pipeline_dxcale(xpath, ypath, geoid_fn, wdir,name = 'gwrDTM_'):
    ti = time.perf_counter()
    start_time = datetime.now()
    """
    Pipeline for downscaling GDEM data using GWR and ensemble methods.
    """
    
    
    expname = os.path.basename(xpath).split(".")[0]
    print(f"Processing {expname}...")
    dw_weighting_list = [0,1,2,3]
    outdir = f"{wdir}/{expname}" #wdir >out_dpath
    os.makedirs(outdir, exist_ok=True)

    #gdemH_fn_tls = ypath.replace(".tif", "_EGM.tif")
    #if not os.path.exists(gdemH_fn_tls):bcor_sub(fine_path=ypath, coarse_path=geoid_fn, output_path=gdemH_fn_tls) # add this into the dataprep 

    opath =  os.path.join(outdir,f'{expname}_{name}.tif')
    overwrite= True#True  # only the 1st time :: fix this also 
    aux_files = True 
    clean = False
    model_out = 1# d:0
    gwrp_fn_list , fmin_fn_list = [],[]
    ti = time.perf_counter()
    for dw_weighting in dw_weighting_list:
        gwrp_fn, fmin_fn = gwrdownxcale(xpath, ypath, opath, geoid_fn, 
                    overwrite=False, oaux=False, epsg_code=4979, clean=True,
                    search_range=0, search_radius=10, 
                    dw_weighting=dw_weighting, dw_idw_power=2.0, dw_bandwidth=1.0,
                    logistic=0, model_out=0, grid_system=None,
                    fmin_run=True)
        gwrp_fn_list.append(gwrp_fn)
        fmin_fn_list.append(fmin_fn)

    assert len(gwrp_fn_list) >= 2, "No file found for Prediction files..."
    assert len(fmin_fn_list) >= 2, "No file found for Prediction files..."
    print('Files check passed!!!')

    init_points, n_iter = 5,15#10,50 #5,15 #10,90 #10,50

    rfile = xpath
    ename_fmin = f"{expname}_{name}_fmin"
    pfiles = fmin_fn_list
    print(f'Running {ename_fmin}')
    avg_raster, opt_raster = ensemble_prediction(ename=ename_fmin, pred_paths=fmin_fn_list, 
                                                ref_path = xpath, 
                                                init_points=init_points, n_iter=n_iter)



    ename_gwrp = f"{expname}_{name}_gwrp"
    pfiles = gwrp_fn_list
    print(f'Running {ename_gwrp}')
    avg_raster, opt_raster = ensemble_prediction(ename=ename_gwrp, pred_paths=gwrp_fn_list, 
                                                ref_path = xpath, 
                                                init_points=init_points, n_iter=n_iter)

    print('*'*32)
    tf = time.perf_counter() - ti
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"stime:{start_time}\netime:{end_time}")
    print(f"runtime:{tf/60}\nduration:{duration}")
    return expname

# cut out OPTe from the code , and also cean up the code [] @write the way as it does not work well 

name = 'gwrDTM_'
outdir = "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/GWRd_dod"
ypaths = [
    "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/GEDI03_fdtm_EGM_1Km.tif",
   # "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/TDEM_DEM_EGM_V_3_GAP_gF3_0_100_inv_dist_dod.tif",
   # "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/TDEM_DEM_EGM_dod.tif",
   # "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/TDEM_DEM_EGM_V_3_GAP_dod.tif",
]

xpaths = [
    "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/TDEM_DEM_EGM_V_3_GAP_gF3_0_100_inv_dist.tif",
    "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/TDEM_DEM_EGM_V_3_GAP.tif",
   # "/media/ljp238/12TBWolf/BRCHIEVE/GDEM/BLOCK/TLS/mosaik/TDEM_DEM_EGM.tif"
]

for ypath in ypaths:
    print('#'*45)
    print(f"Processing {ypath}...")
    bname = os.path.basename(ypath)[:-4]
    print(f"bname: {bname}")
    wdir = f"{outdir}/{bname}"
    os.makedirs(wdir, exist_ok=True)
    
    for xpath in xpaths:
        print('*'*45)
        print(f"Processing {xpath}...")
        # running with None because I am assumiing the geoid has been subtracted already
        expname = pipeline_dxcale(xpath, ypath, geoid_fn=None, wdir=wdir,name = name) 
        fname = f"{name}_{expname}.tif"
        os.system(f'notify-send "{fname} finished" --expire-time=60000')
        with open( 'progress.txt', 'a') as f:
            f.write(f"{fname}\t{datetime.now()}\n")



# want a txt file that writes fname and the time stable so i can track progress 
# and also the time it took to run each f
       
      
