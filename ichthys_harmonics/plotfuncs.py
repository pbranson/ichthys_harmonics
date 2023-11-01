import numpy as np
import matplotlib.pyplot as plt
from ichthys_harmonics.utils import speed, n_speeds


def plot_decomposed_currents(ds, order, mooring, time_pred, ylim=0.6):
    """
    Plot the decomposed tidal currents.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    order : list
        The order of the currents.
    mooring : str
        The mooring location.
    time_pred : array-like
        The prediction times.
    ylim : float, optional
        The y-axis limit. Default is 0.6.

    Returns
    -------
    matplotlib.figure.Figure, array-like
        The figure and axes objects.
    """
    fig, ax = plt.subplots(4,1, figsize=(12,6), gridspec_kw={'hspace':0.1})
    for nco, x in zip(order, ax):
        ds[nco].plot(ax=x, label=nco)
        x.fill_between(time_pred, ds[nco] - ds[f'{nco}_std_err'], ds[nco] + ds[f'{nco}_std_err'], alpha=0.5)
        x.set_xlim([time_pred[0], time_pred[-1]])
        x.set_ylim([-ylim, ylim])
        x.set_xlabel('')
        x.set_ylabel(x.get_ylabel() + '\n[m s$^{-1}$]')
        x.grid()
        if x != ax[-1]:
            x.set_xticklabels([])
    _=ax[0].set_title(f'Decomposed predictions at {mooring}')
    return fig, ax


def plot_combined_currents(ds, order, mooring, time_pred, ylim=0.6):
    """
    Plot the combined tidal currents.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    order : list
        The order of the currents.
    mooring : str
        The mooring location.
    time_pred : array-like
        The prediction times.
    ylim : float, optional
        The y-axis limit. Default is 0.6.

    Returns
    -------
    matplotlib.figure.Figure, array-like
        The figure and axes objects.
    """
    fig, ax = plt.subplots(2,1, figsize=(12,4.5))
    for ntwo, x in zip([[order[0], order[2]], [order[1], order[3]]], ax):
        pred_comb = ds[ntwo[0]] + ds[ntwo[1]]
        pred_err = ds[f'{ntwo[0]}_std_err'] + ds[f'{ntwo[1]}_std_err']
        pred_comb.plot(ax=x)
        x.fill_between(time_pred, pred_comb - pred_err, pred_comb + pred_err, alpha=0.5)
        x.set_xlim([time_pred[0], time_pred[-1]])
        x.set_ylim([-ylim, ylim])
        x.set_xlabel('')
        x.grid()
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('N-S [m s$^{-1}$]')
    ax[1].set_ylabel('E-W [m s$^{-1}$]')
    _=ax[0].set_title(f'Combined predictions at {mooring}')
    return fig, ax


def plot_speed(ds, order, mooring, time_pred, samples=1000, ylim=0.6):
    """
    Plot the total speed.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    order : list
        The order of the currents.
    mooring : str
        The mooring location.
    time_pred : array-like
        The prediction times.
    samples : int, optional
        The number of samples to draw. Default is 1000.
    ylim : float, optional
        The y-axis limit. Default is 0.6.

    Returns
    -------
    matplotlib.figure.Figure, array-like
        The figure and axes objects.
    """
    # Draw from a random distribution n_samp times
    ns_mean = ds[order[0]] + ds[order[2]]
    ns_err = ds[f'{order[0]}_std_err'] + ds[f'{order[2]}_std_err']
    ew_mean = ds[order[1]] + ds[order[3]]
    ew_err = ds[f'{order[1]}_std_err'] + ds[f'{order[3]}_std_err']

    # Get the speed with uncertainty
    speed_all = n_speeds(ns_mean, ns_err, ew_mean, ew_err, n_samples=samples)

    # plot the speed
    fig, ax = plt.subplots(1,1, figsize=(12,4))
    l1 = ax.plot(time_pred, np.mean(speed_all, axis=0), label='w/ noise')
    l2 = ax.plot(time_pred, speed(ns_mean, ew_mean), label='w/o noise')
    ax.fill_between(time_pred, np.percentile(speed_all, 50-34, axis=0), np.percentile(speed_all, 50+34, axis=0), alpha=0.5)
    ax.set_xlim([time_pred[0], time_pred[-1]])
    ax.set_ylim([0, ylim])
    ax.set_xlabel('')
    ax.grid()
    ax.legend(handles=[l1[0], l2[0]])
    ax.set_ylabel('Speed [m s$^{-1}$]')
    _=ax.set_title(f'Total speed predictions at {mooring}')
    return fig, ax