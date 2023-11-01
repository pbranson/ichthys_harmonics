import numpy as np


def pop_timevars(ls_ds):
    """
    Remove time-related variables from the dataset.

    Parameters
    ----------
    ls_ds : xarray.Dataset
        The input dataset.

    Returns
    -------
    xarray.Dataset
        The dataset with time-related variables removed.
    """
    return ls_ds.drop_vars(['h', 'h_predicted', 'h_residual', 'time'])


def murphy_ss(true_signal, fit_signal):
    """
    Calculate the Murphy skill score.

    Parameters
    ----------
    true_signal, fit_signal : array-like
        The true and fitted signals.

    Returns
    -------
    float
        The Murphy skill score.
    """
    return 1 - (np.nanmean((true_signal - fit_signal)**2) / np.nanmean((true_signal - np.nanmean(fit_signal))**2))


def create_time(start, end, step_minutes=10):
    """
    Create a time array.

    Parameters
    ----------
    start, end : datetime-like
        The start and end times.
    step_minutes : int, optional
        The time step in minutes. Default is 10.

    Returns
    -------
    numpy.ndarray
        The time array.
    """
    step = np.timedelta64(step_minutes,'m')
    return np.arange(start, end, step)


def speed(u, v):
    """
    Calculate the speed from the u and v components.

    Parameters
    ----------
    u, v : array-like
        The u and v components.

    Returns
    -------
    numpy.ndarray
        The speed.
    """
    return np.sqrt(u*u + v*v)


def draw_uncertainty(u_mean, u_err, n_samples=1000):
    """
    Draw samples from a normal distribution with mean u_mean and standard deviation u_err.

    Parameters
    ----------
    u_mean, u_err : array-like
        The mean and standard deviation of the normal distribution.
    n_samples : int, optional
        The number of samples to draw. Default is 1000.

    Returns
    -------
    numpy.ndarray
        The drawn samples.
    """
    return np.random.normal(u_mean, u_err, size=(n_samples, len(u_mean)))


def n_speeds(u_mean, u_err, v_mean, v_err, n_samples=1000):
    """
    Calculate the speed from the u and v components with uncertainty.

    Parameters
    ----------
    u_mean, u_err, v_mean, v_err : array-like
        The mean and standard deviation of the u and v components.
    n_samples : int, optional
        The number of samples to draw. Default is 1000.

    Returns
    -------
    numpy.ndarray
        The speed.
    """
    u = draw_uncertainty(u_mean, u_err, n_samples)
    v = draw_uncertainty(v_mean, v_err, n_samples)
    return speed(u, v)