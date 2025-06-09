import os
import time 
import rasterio
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score

def print_context(text):
    print(f'----------- {text} --------------')

def multiple_regression_analysis(
    dependent_path, predictor_paths,
    output_regression_path, output_residuals_path=None,
    include_xy=False, method='forward', p_value=0.05,
    crossval=0, crossval_k=10):
    """
    train model and make prections with linear regresion and two+ features 
    trained_coeffs_path, _ = multiple_regression_analysis(
        dependent_path=uvars.dependent_tif,
        predictor_paths=uvars.predictors_tif_list,
        output_regression_path=uvars.regression_tif,
        output_residuals_path=uvars.residuals_tif,
        include_xy=True,
        method='forward',
        p_value=5,
        crossval=0)

    if trained_coeffs_path:
        pass
    else:
        exit()
    
    """
    
    print_context("stated multiple_regression_analysis...")
    
    ti = time.perf_counter()
    print_context("1. dependent preprocessing ...")
    dependent_arr, meta_for_output = read_rasters([dependent_path])
    dependent_arr = dependent_arr.squeeze()
    print_context("2. covars preprocessing ...")
    predictor_arrs, _ = read_rasters(predictor_paths)

    print_context("3. coords preprocessing ...")
    predictor_names = [f"Predictor_{i}" for i in range(len(predictor_paths))]
    if include_xy:
        fname = 'xy1'
        predictor_names += ['X_coord', 'Y_coord']
    else:
        fname = 'xy0'

    print_context("4. data preprocessing ...")
    X, y, (ys, xs), mask = prepare_data(dependent_arr, predictor_arrs, include_xy=include_xy)

    print_context("5. feature ops selection ...")

    if method == 'forward':
        model, selected_vars, step_info = forward_selection(X, y, p_threshold=p_value / 100.0)
    elif method == 'include_all':
        X_full = sm.add_constant(X)
        model = sm.OLS(y, X_full).fit()
        selected_vars = list(range(X.shape[1]))
        step_info = []
    elif method == 'backward':
        model, selected_vars, step_info = backward_selection(X, y, p_threshold=p_value / 100.0)
    elif method == 'stepwise':
        model, selected_vars, step_info = stepwise_selection(X, y, p_threshold_in=p_value / 100.0)
    else:
        raise NotImplementedError(f"Method '{method}' not implemented. Supported: 'include_all', 'forward', 'backward', 'stepwise'")

    if model is None:
        return None, None
    print_context("6. data modelling ...")
    pred_raster, resid_raster = predict_to_raster(model, selected_vars, X, mask, dependent_arr.shape, y_true=y)

    print_context("7.1 data prediction saving  ...")
    output_regression_path_final = output_regression_path.replace('.tif', f'_{method}_{fname}.tif')
    if output_residuals_path:
        output_residuals_path_final = output_residuals_path.replace('.tif', f'_{method}_{fname}.tif')
    else:
        output_residuals_path_final = None
    
    print_context("7.2 data residual saving  ...")
    save_raster(output_regression_path_final, pred_raster, meta_for_output)
    if output_residuals_path_final is not None:
        save_raster(output_residuals_path_final, resid_raster, meta_for_output)

    print_context("8. model prediction saving  ...")
    coeffs_path = output_regression_path_final.replace('.tif', '_coefficients.csv')
    save_model_info(coeffs_path, model, selected_vars, predictor_names, step_info)

    print_context("8. cross-validation model ...")
    if crossval != 0:
        if crossval == 1:
            cv = LeaveOneOut()
            cvname = 'LeaveOneOut'
        elif crossval == 2:
            cv = KFold(n_splits=2, shuffle=True, random_state=42)
            cvname = 'KFold_2'
        elif crossval == 3:
            cv = KFold(n_splits=crossval_k, shuffle=True, random_state=42)
            cvname = f'KFold_{crossval_k}'
        else:
            raise ValueError("Invalid crossval value")
        print_context(f"8.1 cross-validation {cvname} ...")
        rmse_list, r2_list = [], []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if not selected_vars:
                rmse_list.append(np.nan)
                r2_list.append(np.nan)
                continue

            max_col_idx = X_train.shape[1] - 1
            valid_selected_vars = [v for v in selected_vars if v <= max_col_idx]
            if not valid_selected_vars:
                rmse_list.append(np.nan)
                r2_list.append(np.nan)
                continue

            X_train_sel = sm.add_constant(X_train[:, valid_selected_vars])
            X_test_sel = sm.add_constant(X_test[:, valid_selected_vars])
            
            try:
                cv_model = sm.OLS(y_train, X_train_sel).fit()
                y_pred = cv_model.predict(X_test_sel)
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                r2 = r2_score(y_test, y_pred)
                rmse_list.append(rmse)
                r2_list.append(r2)
            except Exception as e:
                print('error:', e)
                rmse_list.append(np.nan)
                r2_list.append(np.nan)

        cv_df = pd.DataFrame({
            "Fold": list(range(1, len(rmse_list) + 1)),
            "RMSE": rmse_list,
            "R2": r2_list
        })
        print_context("8. cv scores prediction saving  ...")
        cv_path = output_regression_path_final.replace(".tif", f"_{cvname}_cv_scores.csv")
        cv_df.to_csv(cv_path, index=False)
    
    tf = time.perf_counter() - ti 
    print_timing(start_time=tf, end_time=ti, label="[Multi-Regression-Analysis]")

    return coeffs_path, output_regression_path_final


##############################################################
############## DATA OPS
###############################################################

def read_rasters(paths):
    "read and stack rasters"
    arrays = []
    meta = None
    for i, p in enumerate(paths):
        with rasterio.open(p) as src:
            arr = src.read(1, masked=True)
            if i == 0:
                meta = src.meta
            else:
                pass
            arrays.append(arr)
    stacked = np.stack(arrays, axis=-1)
    return stacked, meta

def prepare_data(dependent_arr, predictor_arrs, include_xy, dep_lthrsh=-30, dep_hthresh=1000):
    """ prepara the data and make sure no nulls in the prediction """
    # clean and improve the function 
    # if train data, filter pixels by mask of dependent_arr, if prediction just load 
    if not isinstance(dependent_arr, np.ndarray):
        raise TypeError("dependent_arr must be a numpy array")
    if not isinstance(predictor_arrs, np.ndarray):
        raise TypeError("predictor_arrs must be a numpy array")

    if dependent_arr.ndim != 2:
        raise ValueError(f"dependent_arr must be 2D, got shape {dependent_arr.shape}")
    if predictor_arrs.ndim != 3:
        raise ValueError(f"predictor_arrs must be 3D, got shape {predictor_arrs.shape}")
    if predictor_arrs.shape[0:2] != dependent_arr.shape:
        raise ValueError(f"spatial dimensions of predictor_arrs {predictor_arrs.shape[0:2]} "
                         f"and dependent_arr {dependent_arr.shape} must match")

    dependent_arr1 = dependent_arr.copy()
    del dependent_arr
    dependent_arr1[dependent_arr1 <= dep_lthrsh] = np.nan 
    dependent_arr1[dependent_arr1 >= dep_hthresh] = np.nan
    dependent_arr = dependent_arr1.copy()
    mask = ~np.isnan(dependent_arr)

    for i in range(predictor_arrs.shape[2]):
        pred_band = predictor_arrs[:, :, i]
        pred_valid = ~np.isnan(pred_band)
        mask &= pred_valid

    if mask.ndim != 2 or mask.shape != dependent_arr.shape:
        raise ValueError(f"mask must be 2D with shape {dependent_arr.shape}, got {mask.shape}")

    ys, xs = np.where(mask)

    if len(ys) == 0:
        raise ValueError("No valid data points found after masking")

    y = dependent_arr[ys, xs]

    X = predictor_arrs[ys, xs, :]

    if include_xy:
        coords = np.stack([xs, ys], axis=1)
        X = np.hstack([X, coords])

    return X, y, (ys, xs), mask

##############################################################
############## FEATURE OPS 
###############################################################
# this could go into a separate sctipt and make model agnostic , like RF, DT and other 
def backward_selection(X, y, p_threshold=0.05):
    selected = list(range(X.shape[1]))
    step_info = []

    while selected:
        X_sel = sm.add_constant(X[:, selected])
        model = sm.OLS(y, X_sel).fit()
        pvals = model.pvalues[1:]
        max_pval = pvals.max()
        if max_pval > p_threshold:
            remove_idx = pvals.argmax()
            removed_var = selected[remove_idx]
            selected.remove(removed_var)
            step_info.append({
                "step": len(step_info)+1,
                "removed_variable": removed_var,
                "AIC": model.aic,
                "model_summary": model.summary().as_text()
            })
        else:
            break

    final_model = sm.OLS(y, sm.add_constant(X[:, selected])).fit()
    return final_model, selected, step_info

def stepwise_selection(X, y, p_threshold_in=0.05, p_threshold_out=0.10):
    included = []
    step_info = []
    while True:
        changed = False
        excluded = list(set(range(X.shape[1])) - set(included))
        new_pvals = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[:, included + [new_column]])).fit()
            new_pvals[new_column] = model.pvalues[-1]
        
        if not new_pvals.empty:
            best_pval = new_pvals.min()
            if best_pval < p_threshold_in:
                best_feature = new_pvals.idxmin()
                included.append(best_feature)
                changed = True
                step_info.append({"action": "add", "variable": best_feature, "pval": best_pval})

        if included:
            model = sm.OLS(y, sm.add_constant(X[:, included])).fit()
            pvals = model.pvalues[1:]
            if not pvals.empty:
                worst_pval = pvals.max()
                if worst_pval > p_threshold_out:
                    worst_feature = included[pvals.argmax()]
                    included.remove(worst_feature)
                    changed = True
                    step_info.append({"action": "remove", "variable": worst_feature, "pval": worst_pval})

        if not changed:
            break

    final_model = sm.OLS(y, sm.add_constant(X[:, included])).fit()
    return final_model, included, step_info

def forward_selection(X, y, p_threshold=0.05):
    remaining = list(range(X.shape[1]))
    selected = []
    step_info = []

    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            try:
                try_vars = selected + [candidate]
                X_candidate = sm.add_constant(X[:, try_vars])
                model = sm.OLS(y, X_candidate).fit()
                pvals = model.pvalues[1:]
                max_pval = pvals.max()
                if max_pval < p_threshold:
                    scores_with_candidates.append((model.aic, candidate, model))
            except Exception as e:
                print(f'passing through the Exception ...\n{e}')
                pass
        if not scores_with_candidates:
            break
        scores_with_candidates.sort()
        best_score, best_candidate, best_model = scores_with_candidates[0]
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        step_info.append({
            "step": len(selected),
            "added_variable": best_candidate,
            "AIC": best_score,
            "model_summary": best_model.summary().as_text()
        })
    if selected:
        X_final = sm.add_constant(X[:, selected])
        final_model = sm.OLS(y, X_final).fit()
    else:
        final_model = None
    return final_model, selected, step_info

##############################################################
############## MODEL OPS 
###############################################################
def predict_to_raster(model, selected_vars, X_full, mask, shape, y_true=None):
    pred = np.full(shape, np.nan)
    resid = np.full(shape, np.nan)

    if model is None:
        return pred, resid

    if not selected_vars:
        return pred, resid
    
    max_col_idx = X_full.shape[1] - 1
    valid_selected_vars = [v for v in selected_vars if v <= max_col_idx]
    if not valid_selected_vars:
        return pred, resid

    X_sel = sm.add_constant(X_full[:, valid_selected_vars])
    
    if X_sel.shape[1] != len(model.params):
        if X_sel.shape[1] < len(model.params):
            return pred, resid
        else:
            model_params_adjusted = model.params[:X_sel.shape[1]]
            y_pred = np.dot(X_sel, model_params_adjusted)
    else:
        y_pred = model.predict(X_sel)

    ys, xs = np.where(mask)
    pred[ys, xs] = y_pred

    if y_true is not None and len(y_true) == len(y_pred):
        resid_arr = np.full(shape, np.nan)
        resid_arr[ys, xs] = y_true - y_pred
    else:
        pass

    return pred, resid_arr

def save_raster(path, data, meta):
    meta_out = meta.copy()
    meta_out.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(path, 'w', **meta_out) as dst:
        dst.write(data.astype(rasterio.float32), 1)


def save_model_info(coeffs_path, model, selected_vars, predictor_names, step_info):
    coef_data = {
        "Variable": ["Intercept"] + [predictor_names[i] for i in selected_vars],
        "Coefficient": model.params,
        "P-Value": model.pvalues
    }
    coef_df = pd.DataFrame(coef_data)
    coef_df.to_csv(coeffs_path, index=False)

    steps_path = coeffs_path.replace('.csv', '_steps.csv')
    steps_df = pd.DataFrame(step_info)
    steps_df.to_csv(steps_path, index=False)


##############################################################
############## UTILS OPS 
###############################################################
def print_timing(start_time, end_time, label=""):
    elapsed = end_time - start_time
    print(f"{label}Start time: {time.ctime(start_time)}")
    print(f"{label}End time: {time.ctime(end_time)}")
    print(f"{label}Elapsed time: {elapsed:.2f} sec | {elapsed/60:.2f} min | {elapsed/3600:.2f} hrs | {elapsed/86400:.2f} days")

#[] MAKE IT ALL SKLEARN FRIENLY 