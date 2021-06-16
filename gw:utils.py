from __future__ import division
from __future__ import print_function
import os
import json

import numpy as np
from numpy import inf
import sys
import math

from ..core.utils import (gps_time_to_gmst, ra_dec_to_theta_phi,
                          speed_of_light, logger, run_commandline,
                          check_directory_exists_and_if_not_mkdir)

try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    logger.warning("You do not have gwpy installed currently. You will "
                   " not be able to use some of the prebuilt functions.")

try:
    import lalsimulation as lalsim
except ImportError:
    logger.warning("You do not have lalsuite installed currently. You will"
                   " not be able to use some of the prebuilt functions.")


def asd_from_freq_series(freq_data, df):
    """
    Calculate the ASD from the frequency domain output of gaussian_noise()

    Parameters
    -------
    freq_data: array_like
        Array of complex frequency domain data
    df: float
        Spacing of freq_data, 1/(segment length) used to generate the gaussian noise

    Returns
    -------
    array_like: array of real-valued normalized frequency domain ASD data

    """
    return np.absolute(freq_data) * 2 * df**0.5


def psd_from_freq_series(freq_data, df):
    """
    Calculate the PSD from the frequency domain output of gaussian_noise()
    Calls asd_from_freq_series() and squares the output

    Parameters
    -------
    freq_data: array_like
        Array of complex frequency domain data
    df: float
        Spacing of freq_data, 1/(segment length) used to generate the gaussian noise

    Returns
    -------
    array_like: Real-valued normalized frequency domain PSD data

    """
    return np.power(asd_from_freq_series(freq_data, df), 2)


def time_delay_geocentric(detector1, detector2, ra, dec, time):
    """
    Calculate time delay between two detectors in geocentric coordinates based on XLALArrivaTimeDiff in TimeDelay.c
    Parameters
    -------
    detector1: array_like
        Cartesian coordinate vector for the first detector in the geocentric frame
        generated by the Interferometer class as self.vertex.
    detector2: array_like
        Cartesian coordinate vector for the second detector in the geocentric frame.
        To get time delay from Earth center, use detector2 = np.array([0,0,0])
    ra: float
        Right ascension of the source in radians
    dec: float
        Declination of the source in radians
    time: float
        GPS time in the geocentric frame

    Returns
    -------
    float: Time delay between the two detectors in the geocentric frame

    """
    gmst = gps_time_to_gmst(time)
    theta, phi = ra_dec_to_theta_phi(ra, dec, gmst)
    omega = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    delta_d = detector2 - detector1
    return np.dot(omega, delta_d) / speed_of_light


def get_polarization_tensor(ra, dec, time, psi, mode):
    """
    Calculate the polarization tensor for a given sky location and time

    See Nishizawa et al. (2009) arXiv:0903.0528 for definitions of the polarisation tensors.
    [u, v, w] represent the Earth-frame
    [m, n, omega] represent the wave-frame
    Note: there is a typo in the definition of the wave-frame in Nishizawa et al.
    Parameters
    -------
    ra: float
        right ascension in radians
    dec: float
        declination in radians
    time: float
        geocentric GPS time
    psi: float
        binary polarisation angle counter-clockwise about the direction of propagation
    mode: str
        polarisation mode

    Returns
    -------
    array_like: A 3x3 representation of the polarization_tensor for the specified mode.

    """
    greenwich_mean_sidereal_time = gps_time_to_gmst(time)
    theta, phi = ra_dec_to_theta_phi(ra, dec, greenwich_mean_sidereal_time)
    u = np.array([np.cos(phi) * np.cos(theta), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), 0])
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    if mode.lower() == 'plus':
        return np.einsum('i,j->ij', m, m) - np.einsum('i,j->ij', n, n)
    elif mode.lower() == 'cross':
        return np.einsum('i,j->ij', m, n) + np.einsum('i,j->ij', n, m)
    elif mode.lower() == 'breathing':
        return np.einsum('i,j->ij', m, m) + np.einsum('i,j->ij', n, n)

    omega = np.cross(m, n)
    if mode.lower() == 'longitudinal':
        return np.sqrt(2) * np.einsum('i,j->ij', omega, omega)
    elif mode.lower() == 'x':
        return np.einsum('i,j->ij', m, omega) + np.einsum('i,j->ij', omega, m)
    elif mode.lower() == 'y':
        return np.einsum('i,j->ij', n, omega) + np.einsum('i,j->ij', omega, n)
    else:
        logger.warning("{} not a polarization mode!".format(mode))
        return None


def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordinates in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression

    Parameters
    -------
    latitude: float
        Latitude in radians
    longitude:
        Longitude in radians
    elevation:
        Elevation in meters

    Returns
    -------
    array_like: A 3D representation of the geocentric vertex position

    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis**2 * (semi_major_axis**2 * np.cos(latitude)**2 +
                                   semi_minor_axis**2 * np.sin(latitude)**2)**(-0.5)
    x_comp = (radius + elevation) * np.cos(latitude) * np.cos(longitude)
    y_comp = (radius + elevation) * np.cos(latitude) * np.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * np.sin(latitude)
    return np.array([x_comp, y_comp, z_comp])

def inner_product(aa, bb, frequency, PSD):
    """
    Calculate the inner product defined in the matched filter statistic

    Parameters
    -------
    aa, bb: array_like
        Single-sided Fourier transform, created, e.g., by the nfft function above
    frequency: array_like
        An array of frequencies associated with aa, bb, also returned by nfft
    PSD: bilby.gw.detector.PowerSpectralDensity

    Returns
    -------
    The matched filter inner product for aa and bb

    """
    psd_interp = PSD.power_spectral_density_interpolated(frequency)

    # calculate the inner product
    integrand = np.conj(aa) * bb / psd_interp

    integrand[np.isnan(integrand)] = 0

    df = frequency[1] - frequency[0]
    integral = np.sum(integrand) * df
    return 4. * np.real(integral)

"""
def projection_inner_product(aa, bb, frequency, PSD):

    print('aa is :', aa)
    print("aa type is ", type(aa))
    #print("Size of aa is :", np.size(aa))
    #print("Shape of aa  is :", np.shape(aa))
    #print('Data type of aa  is :', aa.dtype)

    print('bb is :', bb)
    print("bb type is ", type(bb))
    #print("Size of bb is :", np.size(bb))
    #print("Shape of bb  is :", np.shape(bb))
    #print('Data type of bb  is :', bb.dtype)

    #print('Waveform frequency array is : ', frequency)
    #print(type(frequency))
    #print('aa Infinity', np.isinf(aa).any())
    #print('aa nan', np.isnan(aa).any())
    #print('bb Infinity', np.isinf(bb).any())
    #print('bb nan', np.isnan(bb).any())
    #print('frequency Infinity', np.isinf(frequency).any())
    #print('frequency nan ', np.isnan(frequency).any())

    # projection inner product is
    #PSD = bilby.gw.detector.PowerSpectralDensity()
    psd_interp = PSD.power_spectral_density_interpolated(frequency)

    #psd_interp[np.isinf(psd_interp)]
    #print('psd_interp Infinity',np.isinf(psd_interp).any()) ## .any() will check for at least one inf or nan values.
    #print('psd_interp nan ', np.isnan(psd_interp).any())
    #print(type(np.isinf(psd_interp)))

    #print('nan',psd_interp[np.isnan(psd_interp)]) # = 0
    #print('inf',psd_interp[np.isinf(psd_interp)])
    #psd_interp[psd_interp == 'nan+nanj'] = 0

    print('psd_interp is :', psd_interp)
    print("psd_interp type is ", type(psd_interp))
    print("Size of psd_interp is :", np.size(psd_interp))
    print("Shape of psd_interp  is :", np.shape(psd_interp))
    print('Data type of psd_interp  is :', psd_interp.dtype)

    #np.savetxt('./outdir2/psd_interp_data_file',psd_interp,delimiter = ',')

    # calculate the inner product
    integrand = np.conj(aa) * bb / psd_interp

    #print('Integrand Infinity', np.isinf(integrand).any()) ## np.isinf().any()
    #print('Integrand nan ', np.isnan(integrand).any())

    #integrand[np.isinf(integrand)] = 0
    integrand[np.isnan(integrand)] = 0
    #integrand[integrand ==  np.complex(np.nan, np.nan)] = 0
    #np.nan_to_num(integrand)

    print('Integrand is :', integrand)
    print("Integrand  type is ", type(integrand ))
    print("Size of Integrand  is :", np.size(integrand))
    print("Shape of Integrand  is :", np.shape(integrand))
    print('Data type of Integrand  is :', integrand.dtype)

    #print('Integrand Infinity', np.isinf(integrand).any())  ## np.isinf().any()
    #print('Integrand nan ', np.isnan(integrand).any())

    #print('Integrand data type is :', integrand.dtype)
    #np.savetxt('./outdir2/integrand_data_file_1', integrand, delimiter=',')
    #np.savetxt('./outdir2/integrand_data_file', np.array(integrand), delimiter=',')

    df = frequency[1] - frequency[0]

    integral = np.sum(integrand) * df

    result_inner_product = 4. * np.real(integral)
    print("Inner product is :",result_inner_product)
    print('Type of Inner poduct is ', type(result_inner_product))
    print('Data type of result_inner_product  is :', result_inner_product.dtype)
    print("Size of result_inner_product is :", np.size(result_inner_product))
    print("Shape of result_inner_product  is :", np.shape(result_inner_product))

    #np.savetxt('./outdir2/result_inner_product_file', np.array(result_inner_product), delimiter=',')

    return result_inner_product
"""

def noise_weighted_inner_product(aa, bb, power_spectral_density, duration):
    """
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ----------
    aa: array_like
        Array to be complex conjugated
    bb: array_like
        Array not to be complex conjugated
    power_spectral_density: array_like
        Power spectral density of the noise
    duration: float
        duration of the data

    Returns
    ------
    Noise-weighted inner product.

    """

    # print('noise_weighted_inner_product aa is :', aa)
    # print('noise_weighted_inner_product aa has  nan ', np.isnan(aa).any())

    # print('noise_weighted_inner_product bb is :', bb)
    # print('noise_weighted_inner_product bb has  nan ', np.isnan(bb).any())

    # print('noise_weighted_inner_product power_spectral_density is :', power_spectral_density)
    # print('noise_weighted_inner_product Power Spectral Density has  nan ', np.isnan(power_spectral_density).any())

    integrand = np.conj(aa) * bb / power_spectral_density
    # print('noise_weighted_inner_product integrand has  nan ', np.isnan(integrand).any())

    integrand[np.isnan(integrand)] = 0
    # print('After nan = 0, noise_weighted_inner_product integrand has  nan ', np.isnan(integrand).any())

    return 4 / duration * np.sum(integrand)


def matched_filter_snr(signal, frequency_domain_strain, power_spectral_density, duration):
    """
    Calculate the _complex_ matched filter snr of a signal.
    This is <signal|frequency_domain_strain> / optimal_snr

    Parameters
    ----------
    signal: array_like
        Array containing the signal
    frequency_domain_strain: array_like

    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    -------
    float: The matched filter signal to noise ratio squared

    """

    # print('matched_filter_snr Signal is :', signal)
    # print('matched_filter_snr Signal has  nan ', np.isnan(signal).any())

    # print('matched_filter_snr Frequency Domain Strain is :', frequency_domain_strain)
    # print('matched_filter_snr frequency_domain_strain has  nan ', np.isnan(frequency_domain_strain).any())

    # print('matched_filter_snr Power Spectral Density :', power_spectral_density)
    # print('matched_filter_snr Power Spectral Density has  nan ', np.isnan(power_spectral_density).any())

    # print('Duration is :', duration)

    rho_mf = noise_weighted_inner_product(
        aa=signal, bb=frequency_domain_strain,
        power_spectral_density=power_spectral_density, duration=duration)

    # print('matched_filter_snr rho_mf has  nan ', np.isnan(rho_mf).any())

    rho_mf /= optimal_snr_squared(
        signal=signal, power_spectral_density=power_spectral_density,
        duration=duration)**0.5

    # print('optimal_snr rho_mf has  nan ', np.isnan(rho_mf).any())

    return rho_mf


def optimal_snr_squared(signal, power_spectral_density, duration):
    """

    Parameters
    ----------
    signal: array_like
        Array containing the signal
    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    -------
    float: The matched filter signal to noise ratio squared

    """

    # print('optimal_snr Signal is :', signal)
    # print('optimal_snr Signal has  nan ', np.isnan(signal).any())

    # print('optimal_snr Power Spectral Density :', power_spectral_density)
    # print('optimal_snr Power Spectral Density has  nan ', np.isnan(power_spectral_density).any())

    # print('Duration is :', duration)

    return noise_weighted_inner_product(signal, signal, power_spectral_density, duration)


def get_event_time(event):
    """
    Get the merger time for known GW events.

    See https://www.gw-openscience.org/catalog/GWTC-1-confident/html/
    Last update https://arxiv.org/abs/1811.12907:
        GW150914
        GW151012
        GW151226
        GW170104
        GW170608
        GW170729
        GW170809
        GW170814
        GW170817
        GW170818
        GW170823

    Parameters
    ----------
    event: str
        Event descriptor, this can deal with some prefixes, e.g.,
        '151012', 'GW151012', 'LVT151012'

    Returns
    ------
    event_time: float
        Merger time
    """
    event_times = {'150914': 1126259462.4,
                   '151012': 1128678900.4,
                   '151226': 1135136350.6,
                   '170104': 1167559936.6,
                   '170608': 1180922494.5,
                   '170729': 1185389807.3,
                   '170809': 1186302519.8,
                   '170814': 1186741861.5,
                   '170817': 1187008882.4,
                   '170818': 1187058327.1,
                   '170823': 1187529256.5}
    if 'GW' or 'LVT' in event:
        event = event[-6:]

    try:
        event_time = event_times[event[-6:]]
        return event_time
    except KeyError:
        print('Unknown event {}.'.format(event))
        return None


def get_open_strain_data(
        name, start_time, end_time, outdir, cache=False, buffer_time=0, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ----------
    name: str
        The name of the detector to get data for
    start_time, end_time: float
        The GPS time of the start and end of the data
    outdir: str
        The output directory to place data in
    cache: bool
        If true, cache the data
    buffer_time: float
        Time to add to the begining and end of the segment.
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`

    Returns
    -------
    strain: gwpy.timeseries.TimeSeries
        The object containing the strain data. If the connection to the open-data server
        fails, this function retruns `None`.

    """
    filename = '{}/{}_{}_{}.txt'.format(outdir, name, start_time, end_time)

    if buffer_time < 0:
        raise ValueError("buffer_time < 0")
    start_time = start_time - buffer_time
    end_time = end_time + buffer_time

    if os.path.isfile(filename) and cache:
        logger.info('Using cached data from {}'.format(filename))
        strain = TimeSeries.read(filename)
    else:
        logger.info('Fetching open data from {} to {} with buffer time {}'
                    .format(start_time, end_time, buffer_time))
        try:
            strain = TimeSeries.fetch_open_data(name, start_time, end_time, **kwargs)
            logger.info('Saving cache of data to {}'.format(filename))
            strain.write(filename)
        except Exception as e:
            logger.info("Unable to fetch open data, see debug for detailed info")
            logger.info("Call to gwpy.timeseries.TimeSeries.fetch_open_data returned {}"
                        .format(e))
            strain = None

    return strain


def read_frame_file(file_name, start_time, end_time, channel=None, buffer_time=1, **kwargs):
    """ A function which accesses the open strain data

    This uses `gwpy` to download the open data and then saves a cached copy for
    later use

    Parameters
    ----------
    file_name: str
        The name of the frame to read
    start_time, end_time: float
        The GPS time of the start and end of the data
    buffer_time: float
        Read in data with `t1-buffer_time` and `end_time+buffer_time`
    channel: str
        The name of the channel being searched for, some standard channel names are attempted
        if channel is not specified or if specified channel is not found.
    **kwargs:
        Passed to `gwpy.timeseries.TimeSeries.fetch_open_data`

    Returns
    -----------
    strain: gwpy.timeseries.TimeSeries

    """
    loaded = False
    strain = None

    if channel is not None:
        try:
            strain = TimeSeries.read(source=file_name, channel=channel, start=start_time, end=end_time, **kwargs)
            loaded = True
            logger.info('Successfully loaded {}.'.format(channel))
        except RuntimeError:
            logger.warning('Channel {} not found. Trying preset channel names'.format(channel))

    while not loaded:
        ligo_channel_types = ['GDS-CALIB_STRAIN', 'DCS-CALIB_STRAIN_C01', 'DCS-CALIB_STRAIN_C02',
                              'DCH-CLEAN_STRAIN_C02']
        virgo_channel_types = ['Hrec_hoft_V1O2Repro2A_16384Hz', 'FAKE_h_16384Hz_4R']
        channel_types = dict(H1=ligo_channel_types, L1=ligo_channel_types, V1=virgo_channel_types)
        for detector in channel_types.keys():
            for channel_type in channel_types[detector]:
                if loaded:
                    break
                channel = '{}:{}'.format(detector, channel_type)
                try:
                    strain = TimeSeries.read(source=file_name, channel=channel, start=start_time, end=end_time,
                                             **kwargs)
                    loaded = True
                    logger.info('Successfully read strain data for channel {}.'.format(channel))
                except RuntimeError:
                    pass

    if loaded:
        return strain
    else:
        logger.warning('No data loaded.')
        return None


def get_gracedb(gracedb, outdir, duration, calibration, detectors, query_types=None):
    candidate = gracedb_to_json(gracedb, outdir)
    trigger_time = candidate['gpstime']
    gps_start_time = trigger_time - duration
    cache_files = []
    if query_types is None:
        query_types = [None] * len(detectors)
    for i, det in enumerate(detectors):
        output_cache_file = gw_data_find(
            det, gps_start_time, duration, calibration,
            outdir=outdir, query_type=query_types[i])
        cache_files.append(output_cache_file)
    return candidate, cache_files


def gracedb_to_json(gracedb, outdir=None):
    """ Script to download a GraceDB candidate

    Parameters
    ----------
    gracedb: str
        The UID of the GraceDB candidate
    outdir: str, optional
        If given, a string identfying the location in which to store the json
    """
    logger.info(
        'Starting routine to download GraceDb candidate {}'.format(gracedb))
    from ligo.gracedb.rest import GraceDb
    import urllib3

    logger.info('Initialise client and attempt to download')
    try:
        client = GraceDb()
    except FileNotFoundError:
        raise ValueError(
            'Failed to authenticate with gracedb: check your X509 '
            'certificate is accessible and valid')
    try:
        candidate = client.event(gracedb)
        logger.info('Successfully downloaded candidate')
    except urllib3.HTTPError:
        raise ValueError("No candidate found")

    json_output = candidate.json()

    if outdir is not None:
        check_directory_exists_and_if_not_mkdir(outdir)
        outfilepath = os.path.join(outdir, '{}.json'.format(gracedb))
        logger.info('Writing candidate to {}'.format(outfilepath))
        with open(outfilepath, 'w') as outfile:
                json.dump(json_output, outfile, indent=2)

    return json_output


def gw_data_find(observatory, gps_start_time, duration, calibration,
                 outdir='.', query_type=None):
    """ Builds a gw_data_find call and process output

    Parameters
    ----------
    observatory: str, {H1, L1, V1}
        Observatory description
    gps_start_time: float
        The start time in gps to look for data
    duration: int
        The duration (integer) in s
    calibrartion: int {1, 2}
        Use C01 or C02 calibration
    outdir: string
        A path to the directory where output is stored
    query_type: string
        The LDRDataFind query type

    Returns
    -------
    output_cache_file: str
        Path to the output cache file

    """
    logger.info('Building gw_data_find command line')

    observatory_lookup = dict(H1='H', L1='L', V1='V')
    observatory_code = observatory_lookup[observatory]

    if query_type is None:
        logger.warning('No query type provided. This may prevent data from being read.')
        if observatory_code is 'V':
            query_type = 'V1Online'
        else:
            query_type = '{}_HOFT_C0{}'.format(observatory, calibration)

    logger.info('Using LDRDataFind query type {}'.format(query_type))

    cache_file = '{}-{}_CACHE-{}-{}.lcf'.format(
        observatory, query_type, gps_start_time, duration)
    output_cache_file = os.path.join(outdir, cache_file)

    gps_end_time = gps_start_time + duration

    cl_list = ['gw_data_find']
    cl_list.append('--observatory {}'.format(observatory_code))
    cl_list.append('--gps-start-time {}'.format(gps_start_time))
    cl_list.append('--gps-end-time {}'.format(gps_end_time))
    cl_list.append('--type {}'.format(query_type))
    cl_list.append('--output {}'.format(output_cache_file))
    cl_list.append('--url-type file')
    cl_list.append('--lal-cache')
    cl = ' '.join(cl_list)
    run_commandline(cl)
    return output_cache_file


def save_to_fits(posterior, outdir, label):
    """ Generate a fits file from a posterior array """
    from astropy.io import fits
    from astropy.units import pixel
    from astropy.table import Table
    import healpy as hp
    nside = hp.get_nside(posterior)
    npix = hp.nside2npix(nside)
    logger.debug('Generating table')
    m = Table([posterior], names=['PROB'])
    m['PROB'].unit = pixel ** -1

    ordering = 'RING'
    extra_header = [('PIXTYPE', 'HEALPIX',
                     'HEALPIX pixelisation'),
                    ('ORDERING', ordering,
                     'Pixel ordering scheme: RING, NESTED, or NUNIQ'),
                    ('COORDSYS', 'C',
                     'Ecliptic, Galactic or Celestial (equatorial)'),
                    ('NSIDE', hp.npix2nside(npix),
                     'Resolution parameter of HEALPIX'),
                    ('INDXSCHM', 'IMPLICIT',
                     'Indexing: IMPLICIT or EXPLICIT')]

    fname = '{}/{}_{}.fits'.format(outdir, label, nside)
    hdu = fits.table_to_hdu(m)
    hdu.header.extend(extra_header)
    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
    logger.debug('Writing to a fits file')
    hdulist.writeto(fname, overwrite=True)


def plot_skymap(result, center='120d -40d', nside=512):
    """ Generate a sky map from a result """
    import scipy
    from astropy.units import deg
    import healpy as hp
    import ligo.skymap.plot  # noqa
    import matplotlib.pyplot as plt
    logger.debug('Generating skymap')

    logger.debug('Reading in ra and dec, creating kde and converting')
    ra_dec_radians = result.posterior[['ra', 'dec']].values
    kde = scipy.stats.gaussian_kde(ra_dec_radians.T)
    npix = hp.nside2npix(nside)
    ipix = range(npix)
    theta, phi = hp.pix2ang(nside, ipix)
    ra = phi
    dec = 0.5 * np.pi - theta

    logger.debug('Generating posterior')
    post = kde(np.row_stack([ra, dec]))
    post /= np.sum(post * hp.nside2pixarea(nside))

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes([0.05, 0.05, 0.9, 0.9],
                  projection='astro globe',
                  center=center)
    ax.coords.grid(True, linestyle='--')
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_ticks(exclude_overlapping=True, spacing=45 * deg)
    lat.set_ticks(spacing=30 * deg)

    lon.set_major_formatter('dd')
    lat.set_major_formatter('hh')
    lon.set_ticklabel(color='k')
    lat.set_ticklabel(color='k')

    logger.debug('Plotting sky map')
    ax.imshow_hpx(post)

    lon.set_ticks_visible(False)
    lat.set_ticks_visible(False)

    fig.savefig('{}/{}_skymap.png'.format(result.outdir, result.label))


def convert_args_list_to_float(*args_list):
    """ Converts inputs to floats, returns a list in the same order as the input"""
    try:
        args_list = [float(arg) for arg in args_list]
    except ValueError:
        raise ValueError("Unable to convert inputs to floats")
    return args_list


def lalsim_SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase):

    args_list = convert_args_list_to_float(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass_1, mass_2,
        reference_frequency, phase)

    return lalsim.SimInspiralTransformPrecessingNewInitialConditions(*args_list)


def lalsim_GetApproximantFromString(waveform_approximant):
    if isinstance(waveform_approximant, str):
        return lalsim.GetApproximantFromString(waveform_approximant)
    else:
        raise ValueError("waveform_approximant must be of type str")


def lalsim_SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant):

    # Convert values to floats
    [mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
     luminosity_distance, iota, phase, longitude_ascending_nodes,
     eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
     maximum_frequency, reference_frequency] = convert_args_list_to_float(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, iota, phase, longitude_ascending_nodes,
        eccentricity, mean_per_ano, delta_frequency, minimum_frequency,
        maximum_frequency, reference_frequency)

    # Note, this is the approximant number returns by GetApproximantFromString
    if isinstance(approximant, int) is False:
        raise ValueError("approximant not an int")

    return lalsim.SimInspiralChooseFDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, luminosity_distance, iota, phase,
        longitude_ascending_nodes, eccentricity, mean_per_ano, delta_frequency,
        minimum_frequency, maximum_frequency, reference_frequency,
        waveform_dictionary, approximant)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1):
    try:
        lambda_1 = float(lambda_1)
    except ValueError:
        raise ValueError("Unable to convert lambda_1 to float")

    return lalsim.SimInspiralWaveformParamsInsertTidalLambda1(
        waveform_dictionary, lambda_1)


def lalsim_SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2):
    try:
        lambda_2 = float(lambda_2)
    except ValueError:
        raise ValueError("Unable to convert lambda_2 to float")

    return lalsim.SimInspiralWaveformParamsInsertTidalLambda2(
        waveform_dictionary, lambda_2)
