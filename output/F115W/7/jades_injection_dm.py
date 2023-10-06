"""Data model for JADES mock injection recovery tests
"""

import numpy as np
from scipy.stats import truncnorm
from astropy.io import fits
from astropy.wcs import WCS

from collections import OrderedDict

import argparse

# Set up a random number generator for sampling
seed = 90036241
rng = np.random.default_rng(seed=seed)


def create_parser():
    """
    Parse command line arguments
    """

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="User source injection options.")

    parser.add_argument('--outfile', '-o', type=str, default='data/fake_source_cat.fits',
                        help='The path where the output catalog will be saved.')

    parser.add_argument('--clobber', '-c', action='store_true',
        default=False, help='Overwrite the destination file if it already ' +
        'exists?  (default: False)')

    parser.add_argument('--sample_cat', type=str, default=None,
        help='If given, sample some parameters from this real jades catalog')

    parser.add_argument('--n_gals', type=int, default=1,
        help='Number of fake sources')

    parser.add_argument('--field', type=str, default='GOODS-S',
        help='The field to inject fake sources into')

    parser.add_argument('--version', '-f', type=str, default='v0.8',
        help='The version to inject fake sources into')

    return parser


def injection_data_model(bands=[], wcs=None, n_obj=1, col_dtype=np.float64):
    """
    Parameters
    ----------

    bands : sequence of str
        List of filter names

    wcs : astropy.wcs.WCS() instance
        The wcs giving the mapping between celestial coordinates and image
        pixel.  If supplied, it will be added ot the header of the 'POSITION'
        extension.

    n_obj : int
        Number of objects.  Each table will have this length

    Returns
    -------
    injection_data_model : astropy.fits.HDUList
        A list of HDUs, each of which is a binary FITS table, including
        * POSITION - ra, dec
        * SHAPE - profile and orientation
        * FLUX - total fluxes
        * PHYSICAL - physical parameters like redshift, stellar mass
    """

    meta = [("id", np.int64)]
    coldefs = OrderedDict()
    coldefs["POSITION"] = ["ra", "dec", "x_tile", "y_tile"]
    coldefs["SHAPE"] = ["sersic", "rhalf", "q", "pa"]
    coldefs["FLUX"] = bands
    coldefs["PHYSICAL"] = ["redshift", "mstar", "sfr", "met", "a_v"]

    hdul = fits.HDUList([fits.PrimaryHDU()])
    for extn, cols in coldefs.items():
        cd = meta + [(c, col_dtype) for c in cols]
        arr = np.zeros(n_obj, dtype=np.dtype(cd))
        hdul.append(fits.BinTableHDU(arr, name=extn))

    ids = np.arange(1, n_obj+1)
    for hdu in hdul[1:]:
        hdu.data["id"] = ids

    hdul["FLUX"].header["FILTERS"] = ",".join(bands)
    hdul["FLUX"].header["BUNIT"] = "nJy"
    hdul["SHAPE"].header["PA_UNIT"] = "degrees"
    hdul["SHAPE"].header["RH_UNIT"] = "arcsec"
    for hdu in hdul:
        hdu.header["RNG_SEED"] = seed

    if wcs is not None:
        hdul["POSITION"].header.update(wcs.to_header())
        hdul["POSITION"].verify('fix')

    return hdul


def recovery_data_model(inj_dm, append=True,
                        col_dtype=np.float64):
    """Generate a list of HDUs containing information about the recovery of fake
    injected sources.  These HDUs can be returned separately or appended to the
    given injection HDU

    Parameters
    ----------
    inj_dm : astropy.fits.HDUList

    append : optional, default: True
        If True, append the recovery extensions to the injection extension

    Returns
    -------
    hdul :  astropy.fits.HDUList
    """

    n_inj = len(inj_dm[1].data["id"])
    bands = inj_dm["FLUX"].header["FILTERS"].split(",")

    meta = [("id", np.int64)]
    coldefs = OrderedDict()
    coldefs["DETECTION"] = [("detected", bool), ("snr", col_dtype),
                            ("npix", col_dtype), ("wht", col_dtype), ("texp", col_dtype),
                            ("flag", np.uint32)]
    coldefs["PHOTOMETRY"] = ([(b, col_dtype) for b in bands] +
                             [(f"{b}_e", col_dtype) for b in bands])

    if append:
        hdul = inj_dm
    else:
        hdul = fits.HDUList([fits.PrimaryHDU()])

    for extn, cols in coldefs.items():
        cd = meta + cols
        arr = np.zeros(n_inj, dtype=np.dtype(cd))
        arr["id"] = inj_dm[1].data["id"]
        hdul.append(fits.BinTableHDU(arr, name=extn))

    return hdul


def add_header_entry(hdu, key, text):
    """
    Add entry to a header
    """

    hdu.header[key] = text


def populate_from_distributions(icats):
    """
    Parameters
    ----------
    icats : astropy.fits.HDUList
        Injection catalog extensions as an HDUList
    """
    ngals = len(icats[1].data["id"])

    ns_mu, ns_sigma = 1, 0.5
    ns_lower, ns_upper = 0.5, 10

    # Parameters from distributions
    icat = icats["POSITION"]
    icat["ra"] = rng.uniform(low=53.15, high=53.2, size=ngals)
    icat["dec"] = rng.uniform(low=27.75, high=27.8, size=ngals)

    icat = icats["SHAPE"]
    icat["sersic"] = truncnorm((ns_lower - ns_mu) / ns_sigma,
                               (ns_upper - ns_mu) / ns_sigma, loc=ns_mu,
                               scale=ns_sigma).rvs(size=ngals, random_state=seed)
    icat['q']  = np.ones(shape=ngals)
    icat["rhalf"] = -1 * rng.uniform(low=-0.5, high=np.finfo(float).eps, size=ngals)

    # Now do random fluxes with flat slopes in f_nu
    icat = icats["FLUX"]
    filts = icat.header["FILTERS"].split(",")
    norm_flux = rng.uniform(low=10, high=100, size=ngals)
    for f in filts:
        icat[f] = norm_flux

    if "DISTS" in icats:
        dist_hdu = icat["DISTS"]
        add_header_entry(dist_hdu, key='ra', text='-')
        add_header_entry(dist_hdu, key='dec', text='-')
        add_header_entry(dist_hdu, key='pa', text='-')
        add_header_entry(dist_hdu, key='q', text='-')
        add_header_entry(dist_hdu, key='rhalf', text='-')
        add_header_entry(dist_hdu, key='sersic', text='-')

    return icats


def populate_from_catalog(icats, catname=""):
    """
    Parameters
    ----------
    icats : astropy.fits.HDUList
        Injection catalog extensions as an HDUList
    """

    ngals = len(icats[1].data["id"])
    params = sample_from_catalog(catname, ngals)

    icat = icats["SHAPE"]
    icat["q"] = params["Q"]
    krhalf = [k for k in params.keys() if "RHALF" in k.upper()][0]
    icat["rhalf"] = params[krhalf[0]] * 0.03 # convert to arcsec

    icat = icats["FLUX"]
    filts = icats.header["FILTERS"].split(",")
    for f in filts:
        icat[f] = params[f]

    if "DISTS" in icats:
        dist_hdu = icat["DISTS"]
        add_header_entry(dist_hdu, key='q', text='from catalog')
        add_header_entry(dist_hdu, key='rhalf', text='from catalog')

    return icats


def sample_from_catalog(cat_path, ngals, params_to_sample=None):
    """
    Sample parameters from a real JADES catalog
    """
    # Set the default parameters to sample if not provided (axis ratio=Q,
    # half-light radius in F444W=F444W_RHALF, convolved Kron fluxes)
    if params_to_sample is None:
        params_to_sample = ['Q', 'F444W_RHALF', 'KRON_CONV_PHOT']
    params = dict.fromkeys(params_to_sample)

    hdul = fits.open(cat_path)
    all_filts = hdul["FILTERS"].data["NAME"]

    # Sample indices with replacement from the catalog
    inds = rng.choice(len(hdul['FLAG'].data), size=ngals)
    if len(inds) > 0:
        for param in params_to_sample:
            if (('RHALF' in param.upper()) or
                (param.upper() in ['RA', 'DEC', 'Q'])):
                params[param] = hdul['SIZE'].data[param.upper()][inds]
            elif 'PHOT' in param:
                hdu_name = re.sub('\d', '', param.split('_PHOT')[0])
                hdu = hdul[hdu_name]
                all_filts = np.unique([name.split('_')[0] for name in
                                       hdu.data.names])
                for filt in all_filts:
                    if 'F' in filt:
                        key = f'{filt}_{param.split("_CONV")[0].split("_PHOT")[0]}'
                    else:
                        continue
                    params[filt] = hdu.data[key][inds]
    hdul.close()

    return params


def make_and_populate(args):

    ngals = args.ngals

    # Set up the HDUs for the metadata, containing the WCS information and the
    # distributions from which the parameters for the fake sources were drawn
    # Grab the WCS from the v0.8 F444W GOODS-S JADES mosaic for now
    mosaic_path = f"/data/groups/comp-astro/jades/jades-data/{args.field}/images/JWST/JADES/{args.version}/F444W/mosaic_F444W.fits"
    wcs = WCS(fits.getheader(mosaic_path, "SCI"))

    # The filters for which to have photometry
    filts = ['F070W', 'F090W', 'F115W', 'F150W', 'F200W',
             'F277W', 'F335M', 'F356W', 'F410M', 'F444W']

    # Set up to make the galsim extensions of the injection catalog
    objids = np.arange(1, ngals+1, dtype=int)
    icats = injection_data_model(bands=filts, wcs=wcs, n_obj=ngals)
    for icat in icats[1:]:
        icat.data["id"] = objids  # this is already done in injection_data_model, but lets be explicit

    # add object parameters
    icats = populate_from_distributions(icats)
    if args.sample_cat:
        icats = populate_from_catalog(icats, args.sample_cat)
    icats.writeto(args.outfile, overwrite=args.clobber)


def make_example(args):

    # Set up the HDUs for the metadata, containing the WCS information and the
    # distributions from which the parameters for the fake sources were drawn
    # Grab the WCS from the v0.8 F444W GOODS-S JADES mosaic for now
    try:
        mosaic_path = f"/data/groups/comp-astro/jades/jades-data/{args.field}/images/JWST/JADES/{args.version}/F444W/mosaic_F444W.fits"
        wcs = WCS(fits.getheader(mosaic_path, "SCI"))
    except(OSError, FileNotFoundError):
        wcs = None

    # The filters for which to have photometry
    filts = ['F070W', 'F090W', 'F115W', 'F150W', 'F200W',
             'F277W', 'F335M', 'F356W', 'F410M', 'F444W']
    # Set up to make the galsim extensions of the injection catalog
    ngals = 1  # The number of fake sources to inject
    objids = np.arange(1, ngals+1, dtype=int)
    icats = injection_data_model(bands=filts, wcs=wcs, n_obj=ngals)
    for icat in icats[1:]:
        icat.data["id"] = objids  # this is already done in injection_data_model, but lets be explicit

    ircats = recovery_data_model(icats, append=True)
    ircats.writeto(args.outfile, overwrite=args.clobber)


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    make_example(args)
