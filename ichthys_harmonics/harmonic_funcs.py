import numpy as np
import pandas as pd
import xarray as xr
import pyTMD

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


def load_constituents_names(c=None):
    """
    Load tidal constituents names from default lists.
    
    Parameters
    ----------
    c : str, optional
        The tidal constituents list to load. If None, the default FES2014
        constituents are loaded. Should accept any of the following:
            'default_fes_constituents'
            'semidiurnal'
            'diurnal'
            'major_semidiurnal'
            'major_diurnal'
    
    Returns
    -------
    list
        The names of the specified tidal constituents. If `c` is a
        string, a list is returned. If `c` is a list of strings, a
        list of lists is returned with one list per constituent.

    """
    if c is None:
        return default_fes_constituents
    elif isinstance(c, str) & (c in locals()):
        return eval(c)
    else:
        local_vars = np.array(list(locals().keys()).remove('cpd'))
        raise ValueError(
            "c must be None or one of the following: "
            + ", ".join(local_vars)
        )


def load_constituents_properties(c=None):
    """
    Load tidal constituents properties from pyTMD.

    Parameters
    ----------
    c : str or list of str, optional
        The tidal constituents to load. If None, the default FES2014
        constituents are loaded. If a list of strings, the properties of
        each constituent are loaded and returned as a DataFrame.
        Should accept any of the following:
            'default_fes_constituents'
            'semidiurnal'
            'diurnal'
            'major_semidiurnal'
            'major_diurnal'

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The properties of the specified tidal constituents. If `c` is a
        string, a Series is returned. If `c` is a list of strings, a
        DataFrame is returned with one row per constituent and columns
        for the constituent properties.

    """
    if isinstance(c, list):
        df = pd.DataFrame({_c: load_constituents_properties(_c) for _c in c}).T
        df = df.sort_values("omega")
        return df
    
    if (c is None) | (isinstance(c, str) & (c in locals())):
        df = load_constituents_properties(load_constituents_names(c))
        return df
    
    # Main output of the function
    elif isinstance(c, str) & (c not in locals()):
        p_names = ["amplitude", "phase", "omega", "alpha", "species"]
        p = pyTMD.load_constituent(c)
        s = pd.Series({_n: _p for _n, _p in zip(p_names, p)})
        s["omega_cpd"] = s["omega"] * cpd
        return s


def get_tidal_arguments(time, constits=None):
    """
    Extract the time-varying tidal constituent properties for the input time series.

    Parameters
    ----------
    time : array-like
        The time argument to convert. If a dict, it should contain
        keyword arguments that can be passed to pandas.date_range. If
        an array-like, it should be convertible to a pandas DatetimeIndex.
    constits : list of str, optional
        The tidal constituents to load. If None, the default FES2014
        constituents are loaded.
        
    Returns
    -------
    xarray.Dataset
        A dataset indexed by the input time and constituents chosen in 
        'constits' variable.

    """
    # load tidal constituent properties
    eq = load_constituents_properties(constits)

    if isinstance(time, dict):
        time = pd.date_range(**time)
    elif not isinstance(time, pd.DatetimeIndex):
        # convert to DatetimeIndex
        time = pd.DatetimeIndex(time)

    # convert from calendar date to days relative to Jan 1, 1992 (48622 MJD)
    tide_time = pyTMD.time.convert_calendar_dates(
        time.year,
        time.month,
        time.day,
        hour=time.hour,
        minute=time.minute,
    )

    # delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(["data", "merged_deltat.data"])
    deltat = pyTMD.time.interpolate_delta_time(delta_file, tide_time)

    # Extract the nodal corrections
    pu, pf, G = pyTMD.arguments(
        tide_time + 48622.0,
        eq.index.values,
        deltat=deltat,
        corrections="FES",  # model_format='FES'
    )
    # pu, pf are nodal corrections
    # G is the equilibrium argument and common to all models

    # Initiate the dataset
    ds = xr.Dataset(
        None,
        coords=dict(
            time=("time", time), constituents=("constituents", eq.index.values)
        ),
    )

    # Add the constituent corrections as variables
    ds["pu"] = (("time", "constituents"), pu)
    ds["pf"] = (("time", "constituents"), pf)
    ds["G"] = (("time", "constituents"), G)

    # Add the constituent properties as variables
    params = eq.to_xarray().rename(index="constituents")
    ds = xr.merge([ds, params])

    # complex argument
    ds["th"] = ds.G * np.pi / 180.0 + ds.pu

    return ds


def _get_tidal_arguments_float(time, csts_float):
    """
    Load tidal arguments with frequency specified as float
    
    """
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


def get_base_sinusoids(time, constituents):
    """
    Calculate the base sinusoids for a set of tidal constituents.

    Parameters
    ----------
    time : array-like or dict
        The time values to calculate the base sinusoids for. If a dict,
        it should contain keyword arguments that can be passed to
        pandas.date_range. If an array-like, it should be convertible
        to a pandas DatetimeIndex.
    constituents : list of str
        The tidal constituents to use.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the base sinusoids for the specified time
        values and tidal constituents. The columns are the tidal
        constituents and the index is the time values.

    """
    # split constituents from frequencies
    csts_string = [c for c in constituents if isinstance(c, str)]
    csts_float = [c for c in constituents if isinstance(c, float)]

    # get named tidal arguments
    if len(csts_string)>0:
        ta = get_tidal_arguments(time).sel(constituents=csts_string)
    else:
        # add dummy constituent
        ta = get_tidal_arguments(time).sel(constituents=["m2"])

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


def rotate_2D(X, Y, theta):
    """
    Perform a simple 2D rotation (copied from afloat package for now).

    This function rotates the input 2D data (X, Y) by the specified angle theta.

    Parameters
    ----------
    X, Y : array-like
        The input 2D data.
    theta : float
        The rotation angle in radians.

    Returns
    -------
    list
        The rotated 2D data [X1, X2].
    """
    # Perform the rotation
    X1 = X * np.cos(theta) - Y * np.sin(theta)
    X2 = X * np.sin(theta) + Y * np.cos(theta)

    return [X1, X2]


def pca_angle(x_data, y_data):
    """
    Principal component analysis of x_data and y_data (copied from afloat package for now).

    This function calculates the angle of the first principal component of the input data.

    This can be done nicely as one big matrix, see:
    https://medium.com/@nahmed3536/a-python-implementation-of-pca-with-numpy-1bbd3b21de2e

    Parameters
    ----------
    x_data, y_data : array-like
        The input data.

    Returns
    -------
    float
        The angle of the first principal component.
    """
    # Standardize the input data
    std_x = (x_data - x_data.mean()) / x_data.std()
    std_y = (y_data - y_data.mean()) / y_data.std()

    # Calculate the covariance matrix
    sig = np.cov(x_data, y_data)

    # Calculate the eigenvalues and eigenvectors
    w, v = np.linalg.eig(sig)

    # Get the first eigenvector
    v1 = v[0, :] 

    # Calculate the angle of the first eigenvector
    theta_e = np.arctan2(v1[1], v1[0])

    # Return the negative of the angle (convention)
    return -theta_e 

# Rotate data
def pca_rotate_data(xr_data, yr_data):
    """
    Rotate data to principal component direction.

    This function rotates the input data to the direction of the first principal component.

    Parameters
    ----------
    xr_data, yr_data : array-like
        The input data.

    Returns
    -------
    array-like, array-like, float
        The rotated data and the rotation angle.
    """
    # Calculate the PCA angle
    theta = pca_angle(xr_data, yr_data)

    # Rotate the data
    U_rot, V_rot = rotate_2D(xr_data, yr_data, -theta)

    return U_rot, V_rot, theta


def rotate_predictions(ds):
    """
    Rotate predictions to principal component direction.

    This function rotates the predictions in the input dataset to the direction of the first principal component.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.

    Returns
    -------
    xarray.Dataset
        The dataset with rotated predictions.
    """
    # Rotate the predictions
    ds['BTEW'], ds['BTNS'] = rotate_2D(ds['BTEW'], ds['BTNS'], ds.attrs['BTEW_theta'])
    ds['ITEW'], ds['ITNS'] = rotate_2D(ds['ITEW'], ds['ITNS'], ds.attrs['ITEW_theta'])
    return ds