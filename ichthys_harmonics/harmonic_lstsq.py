import os
import glob
import numpy as np
import xarray as xr
from .utils import murphy_ss
from .harmonic_funcs import get_base_sinusoids, load_constituents_names, rotate_2D


# ---------------------------  least square ------------------------------------


def hanalysis_least_square(y, t, constituents, prefix="h"):
    """
    Perform harmonic analysis using least squares.

    Parameters
    ----------
    y : array-like or xarray.DataArray
        The time series to analyze.
    t : array-like or xarray.DataArray
        The time values corresponding to `y`.
    constituents : list of str
        The tidal constituents to use in the analysis.
    prefix : str, optional
        The prefix to use for the output variable names.

    Returns
    -------
    xarray.Dataset
        A dataset containing the results of the harmonic analysis.

    """
    # Convert input arrays to numpy arrays if they are xarray DataArrays
    if isinstance(y, xr.DataArray):
        y = y.values.squeeze()
    if isinstance(t, xr.DataArray):
        t = t.values

    n = y.size
    m = len(constituents)

    # Get base tidal sinusoidal timeseries
    bcos, bsin = get_base_sinusoids(t, constituents)

    # Perform least squares analysis
    X = np.hstack([bcos, bsin])
    C = np.linalg.inv(X.T @ X)
    h = C @ X.T @ y

    # Calculate uncertainties
    yh = X @ h  # predicted values
    S = (y - yh).T @ (y - yh)  # sum of squared residual
    sigma_h2 = S / (n - m)
    h_var = sigma_h2 * np.diag(C)
    cov = sigma_h2 * C

    # Create output dataset
    return xr.Dataset(
        {
            prefix + "_real": ("c", h[:m]),
            prefix + "_imag": ("c", h[m:]),
            prefix + "_real_var": ("c", h_var[:m]),
            prefix + "_imag_var": ("c", h_var[m:]),
            prefix + "_cov": (["coef","coef"], cov),
            prefix : ("time", y),
            prefix + "_predicted": ("time", yh),
            prefix + "_residual": ("time", y-yh),
        },
        coords=dict(c=constituents,
                    time=t,
                    coef=np.arange(2*len(constituents))
                    ),
    )    


def harmonic_fit(time, obs, constituents=None, ds_attrs=None, verbose=True):
    """
    Fit a set of tidal constituents to a time series using least squares.

    Parameters
    ----------
    time : array-like or xarray.DataArray
        The time values corresponding to the observations.
    obs : array-like or xarray.DataArray
        The observations to fit.
    ds_attrs : dict
        A dictionary of attributes to add to the output dataset.

    Returns
    -------
    xarray.Dataset
        A dataset containing the results of the harmonic fit.

    """    
    # Convert input arrays to numpy arrays if they are xarray DataArrays
    if isinstance(obs, xr.DataArray):
        obs = obs.values.squeeze()
    if isinstance(time, xr.DataArray):
        time = time.values
        
    # get the tidal constituents
    constit_names = load_constituents_names(constituents)

    # actually perform the analysis
    ha_ls = hanalysis_least_square(obs, time, constit_names)
    if verbose:
        skill_score = np.round(100 * murphy_ss(obs, ha_ls['h_predicted']), 2)
        print(f"Skill score: {skill_score:.2f}%")

    # Add the data attrs
    if ds_attrs is not None:
        ha_ls.attrs.update(ds_attrs)

    # Add the stddev of the residuals
    ha_ls.attrs['stddev'] = np.std(ha_ls['h_residual'].values)

    return ha_ls


def load_tide_predictions(time, nc_dir, mooring, order):
    """
    Load tidal predictions from netCDF files and perform least squares prediction.

    Parameters
    ----------
    time : array-like or xarray.DataArray
        The time values to predict.
    nc_dir : str
        The path to the directory containing the netCDF files.
    mooring : str
        The name of the mooring.
    order : list of str
        The tidal constituents to predict.

    Returns
    -------
    xarray.Dataset
        A dataset containing the predicted tidal values and standard errors.

    """
    # Initialise the dataset
    ds_pred = xr.Dataset(coords={'time': time})

    for nco in order:
        # Load the netcdf file
        nc = glob.glob(os.path.join(nc_dir, f'*{mooring}*{nco}*.nc'))[0]
        ds = xr.open_dataset(nc)

        # Do the prediction
        yh, std_err = least_squares_prediction(time, ds)

        # Add to the dataset
        ds_pred[nco] = xr.DataArray(yh, dims=['time'])
        ds_pred[f'{nco}_std_err'] = xr.DataArray(std_err + ds.attrs['stddev'], dims=['time'])
        ds_pred.attrs[f'{nco}_theta'] = ds.attrs['pca_angle']
        
    return ds_pred


def least_squares_prediction(time, ha_ls):
    """
    Perform least squares prediction of tidal values.

    Parameters
    ----------
    time : array-like or xarray.DataArray
        The time values to predict.
    ha_ls : xarray.Dataset
        The dataset containing the harmonic analysis results.

    Returns
    -------
    yh : array-like
        The predicted values.
    """    
    # Get the base sinusoids with nodal corrections
    bcos, bsin = get_base_sinusoids(time, ha_ls.c.values)
    X = np.hstack([bcos, bsin])
    m = len(ha_ls.c.values)
    hp = np.hstack([ha_ls.h_real.values, ha_ls.h_imag.values])

    # Do the prediction
    yh = X @ hp

    # Get the UQ
    if 'h_cov' in ha_ls:
        std_err = np.sqrt(np.sum((X @ ha_ls.h_cov.values) * X, axis=1))
    else:
        std_err = None
    
    return yh, std_err
    

def print_comb_skillscores(ds, ha_ls_BTNS, ha_ls_BTEW, ha_ls_ITNS, ha_ls_ITEW):
    """
    Print combined skill scores.

    This function calculates and prints the combined skill scores for North-South, East-West, and total speed.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the measured values and time values.
    ha_ls_BTNS, ha_ls_BTEW, ha_ls_ITNS, ha_ls_ITEW : xarray.Dataset
        The datasets containing the least squares harmonic analysis results.

    Returns
    -------
    None
    """
    # Identify non-NaN values
    nanx = ~np.isnan(ds['v_meas'].values)
    
    # Initialize total North-South and East-West fits
    total_nsfit = np.zeros_like(ds['v_meas'].values[nanx])
    total_ewfit = np.zeros_like(ds['u_meas'].values[nanx])
    
    # Loop over North-South and East-West datasets
    for vt, ut in zip([ha_ls_BTNS, ha_ls_ITNS], [ha_ls_BTEW, ha_ls_ITEW]):
        # Perform least squares prediction
        v_fit, _ = least_squares_prediction(ds['time'].values[nanx], vt)
        u_fit, _ = least_squares_prediction(ds['time'].values[nanx], ut)
        
        # Rotate the fits
        ew_fit, ns_fit = rotate_2D(u_fit, v_fit, ut.attrs['pca_angle'])
        
        # Add to total fits
        total_nsfit += ns_fit
        total_ewfit += ew_fit

    # Print combined skill scores
    print(f"North-South combined score: {100*murphy_ss(ds['v_meas'].values[nanx], total_nsfit):.2f}%")
    print(f"East-West combined score: {100*murphy_ss(ds['u_meas'].values[nanx], total_ewfit):.2f}%")
    print(f"Total speed score: {100*murphy_ss(np.sqrt(ds['v_meas'].values[nanx]**2 + ds['u_meas'].values[nanx]**2), np.sqrt(total_nsfit**2 + total_ewfit**2)):.2f}%")