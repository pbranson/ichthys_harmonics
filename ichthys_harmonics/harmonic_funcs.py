import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import pyTMD
from .utils import murphy_ss, create_time


default_fes_constituents = [
    "2n2",
    "eps2",
    "j1",
    "k1",
    "k2",
    "l2",
    "lambda2",
    "m2",
    "m3",
    "m4",
    "m6",
    "m8",
    "mf",
    "mks2",
    "mm",
    "mn4",
    "ms4",
    "msf",
    "msqm",
    "mtm",
    "mu2",
    "n2",
    "n4",
    "nu2",
    "o1",
    "p1",
    "q1",
    "r2",
    "s1",
    "s2",
    "s4",
    "sa",
    "ssa",
    "t2",
]
semidiurnal = ["m2", "s2", "n2", "k2", "nu2", "mu2", "l2", "t2", "2n2"]
diurnal = ["k1", "o1", "p1", "q1", "j1"]
major_semidiurnal = ["m2", "s2"]
major_diurnal = ["k1", "o1"]

cpd = 86400 / 2 / np.pi


def load_constituents_properties(c=None):
    if c is None:
        c = default_fes_constituents
    if isinstance(c, list):
        df = pd.DataFrame({_c: load_constituents_properties(_c) for _c in c}).T
        df = df.sort_values("omega")
        return df
    elif isinstance(c, str):
        p_names = ["amplitude", "phase", "omega", "alpha", "species"]
        p = pyTMD.load_constituent(c)
        s = pd.Series({_n: _p for _n, _p in zip(p_names, p)})
        s["omega_cpd"] = s["omega"] * cpd
        return s

eq = load_constituents_properties(default_fes_constituents)


def load_constituents(constituents=None):
    if constituents is not None:
        c = constituents
    else:
        c = default_fes_constituents
    return c


def get_tidal_arguments(time):

    if isinstance(time, dict):
        time = pd.date_range(**time)
    elif not isinstance(time, pd.DatetimeIndex):
        # convert to DatetimeIndex
        time = pd.DatetimeIndex(time)

    # -- convert from calendar date to days relative to Jan 1, 1992 (48622 MJD)
    tide_time = pyTMD.time.convert_calendar_dates(
        time.year,
        time.month,
        time.day,
        hour=time.hour,
        minute=time.minute,
    )

    # -- delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(["data", "merged_deltat.data"])
    deltat = pyTMD.time.interpolate_delta_time(delta_file, tide_time)

    pu, pf, G = pyTMD.arguments(
        tide_time + 48622.0,
        default_fes_constituents,
        deltat=deltat,
        corrections="FES",  # model_format='FES'
    )
    # pu, pf are nodal corrections
    # G is the equilibrium argument and common to all models
    ds = xr.Dataset(
        None,
        coords=dict(
            time=("time", time), constituents=("constituents", default_fes_constituents)
        ),
    )
    ds["pu"] = (("time", "constituents"), pu)
    ds["pf"] = (("time", "constituents"), pf)
    ds["G"] = (("time", "constituents"), G)

    params = eq.to_xarray().rename(index="constituents")
    ds = xr.merge([ds, params])

    # complex argument
    ds["th"] = ds.G * np.pi / 180.0 + ds.pu

    return ds


def get_base_sinusoids(t, constituents):

    # split constituents from frequencies
    csts_string = [c for c in constituents if isinstance(c, str)]
    csts_float = [c for c in constituents if isinstance(c, float)]

    # get named tidal arguments
    if len(csts_string)>0:
        ta = get_tidal_arguments(t).sel(constituents=csts_string)
    else:
        # add dummy constituent
        ta = get_tidal_arguments(t).sel(constituents=["m2"])

    # get flow tidal arguments on matching timeline
    if len(csts_float)>0:
        ta_float = _get_tidal_arguments_float(ta.time, csts_float)
        # merge all
        ta = xr.merge([ta, ta_float])
        if len(csts_string)==0:
            # get rid of dummy constituent
            ta = ta.sel(constituents=[c for c in list(ta.constituents.values) if c!="m2"])

    # assemble nodally adjusted base sinusoids
    bcos = ta.pf * np.cos(ta.th)
    bsin = -ta.pf * np.sin(ta.th)

    return bcos.values, bsin.values


def _get_tidal_arguments_float(time, csts_float):
    """load tidal arguments with frequency specified as float"""
    ta = xr.Dataset(
        dict(
            pu=(("time", "constituents"), np.zeros((time.size, len(csts_float)))),
            pf=(("time", "constituents"), np.ones((time.size, len(csts_float)))),
            omega_cpd=("constituents", csts_float),
        ),
        coords=dict(
            constituents=("constituents", [str(c) for c in csts_float]),
            time=time,
        ),
    )
    ta["omega"] = ta["omega_cpd"] / cpd
    ta["th"] = ta["omega"] * ((time - time[0]) / pd.Timedelta("1s"))
    return ta


# ---------------------------  least square ------------------------------------


def hanalysis_least_square(y, t, constituents, prefix="h"):

    if isinstance(y, xr.DataArray):
        y = y.values.squeeze()
    if isinstance(t, xr.DataArray):
        t = t.values

    n = y.size
    m = len(constituents)

    # get base tidal sinusoidal timeseries
    bcos, bsin = get_base_sinusoids(t, constituents)

    # least square
    X = np.hstack([bcos, bsin])
    C = np.linalg.inv(X.T @ X)
    h = C @ X.T @ y

    # toward getting uncertainties
    yh = X @ h  # predicted values
    S = (y - yh).T @ (y - yh)  # sum of squared residual
    sigma_h2 = S / (n - m)
    h_var = sigma_h2 * np.diag(C)
    cov = sigma_h2 * C

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


def harmonic_fit(time, obs, ds_attrs):
    c = load_constituents()
    
    # get the tidal arguments (equilibrium phase, nodal corrections ...)
    ar = get_tidal_arguments(time)

    # actually perform the analysis
    ha_ls = hanalysis_least_square(obs, time, ar.constituents.values)
    print(np.round(100*murphy_ss(obs, ha_ls['h_predicted']), 2))

    # add the data attrs
    ha_ls.attrs = ds_attrs

    # add the stddev of the residuals
    ha_ls.attrs['stddev'] = np.std(ha_ls['h_residual'].values)

    return ha_ls


def load_tide_predictions(time, nc_dir, mooring, order):

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
        
    return ds_pred


def least_squares_prediction(time, ha_ls):
    
    # Do the prediction
    bcos, bsin = get_base_sinusoids(time, ha_ls.c.values)
    X = np.hstack([bcos, bsin])
    m = len(ha_ls.c.values)
    hp = np.hstack([ha_ls.h_real.values, ha_ls.h_imag.values])

    yh = X @ hp
    if 'h_cov' in ha_ls:
        std_err = np.sqrt(np.sum((X @ ha_ls.h_cov.values) * X, axis=1))
    else:
        std_err = None
    
    return yh, std_err
    