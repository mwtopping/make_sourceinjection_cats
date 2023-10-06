  #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_test_image.py -
Use galsim to generate some images that can be used for training
and testing. Largely adopated from Ben Johnson's make_test_image.py.
These will be single band images. The images include
   * galaxy_grid - A small grid of galaxies with varied shape parameters
                   (size, sersic, axis ratio) and at different S/N
"""

#psf = galsim.InterpolatedImage(galsim.Image(psf), wcs=local_wcs)
#gal = galsim.Convolve(psf, gal)
#https://galsim-developers.github.io/GalSim/_build/html/arbitrary.html
#https://galsim-developers.github.io/GalSim/_build/html/image_class.html#galsim.Image

#Filename: injection_data_model.fits
#No.    Name      Ver    Type      Cards   Dimensions   Format
#  0  PRIMARY       1 PrimaryHDU       4   ()
#  1  POSITION      1 BinTableHDU     15   1R x 3C   [K, D, D]
#  2  SHAPE         1 BinTableHDU     21   1R x 5C   [K, D, D, D, D]
#  3  FLUX          1 BinTableHDU     33   1R x 11C   [K, D, D, D, D, D, D, D, D, D, D]
#  4  PHYSICAL      1 BinTableHDU     21   1R x 6C   [K, D, D, D, D, D]
#>>> hdu['POSITION'].data.names
#['id', 'ra', 'dec']
#>>> hdu['SHAPE'].data.names
#['id', 'sersic', 'rhalf', 'q', 'pa']
#>>> hdu['FLUX'].data.names
#['id', 'F070W', 'F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F335M', 'F356W', 'F410M', 'F444W']
#>>> hdu['PHYSICAL'].data.names
#['id', 'redshift', 'mstar', 'sfr', 'met', 'a_v']

import sys
import argparse
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import galsim
import time





#########################################
# Routine to parse command line arguments
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Create a simulated image with sources for injection into JADES mosaics.")

    parser.add_argument("--cat",
        type=str,
        default='input_catalog.fits',
        help='Input catalog detailing the injected source population.')

    parser.add_argument("--header",
        type=str,
        default='image_header.txt',
        help='The header for the output image with injected sources.')

    parser.add_argument("--output",
        type=str,
        default='output_injection_image.fits',
        help='Output file with the injected sources and catalog.')

    parser.add_argument("--band",
        type=str,
        default='F200w',
        help='Image band.')

    parser.add_argument("--origin",
        type=int,
        default=0,
        help='image origin (0 or 1).')

    parser.add_argument("--seed",
        type=int,
        default=0,
        help='RNG seed.')

    parser.add_argument("--id_fac",
        type=int,
        default=1000,
        help='ID prefix multiplied (x seed).')

    parser.add_argument("--psf",
        type=str,
        default='mpsf_F200W.fits',
        help='Input PSF fits file.')

    parser.add_argument("--data_model",
        type=str,
        default='injection+recovery_data_model.fits',
        help='Data model FITS for injection catalog.')

    parser.add_argument('-v', '--verbose',
                dest='verbose',
                action='store_true',
                help='Print helpful information to the screen? (default: False)',
                default=False)

    return parser

#########################################
# Refine catalog
#########################################
def refine_catalog(input_catalog_hdu, header, band='F200W', seed=0, data_model='data_model.fits'):

  rng = np.random.default_rng(seed=seed)

  #position angle
  idx    = input_catalog_hdu['shape'].data['ID']
  pa     = input_catalog_hdu['shape'].data['pa']
  q      = input_catalog_hdu['shape'].data['q']
  rhalf  = input_catalog_hdu['shape'].data['rhalf']
  sersic = input_catalog_hdu['shape'].data['sersic']
  flux   = input_catalog_hdu['flux'].data
  #physical = input_catalog_hdu['physical'].data
  #physical = input_catalog_hdu['SOURCE_PROPS'].data
  n      = len(idx)
  print(f'n = {n}')


  print(f'Rhalf stats min {np.nanmin(rhalf)} std {np.nanstd(rhalf)} median {np.nanmedian(rhalf)} mean {np.nanmean(rhalf)} max {np.nanmax(rhalf)}')

  xi = np.where(
       (np.isnan(idx)==False)&
       (np.isnan(pa)==False)&
       (np.isnan(q)==False)&
       (np.isnan(rhalf)==False)&
       (np.isnan(sersic)==False)&
       (sersic>0.3)&
       (sersic<6.2)&
       (np.isnan(flux[band])==False))[0]
  nxi = len(xi)

  template_hdu  = fits.open(data_model)
  position_hdu  = template_hdu['POSITION']
  shape_hdu     = template_hdu['SHAPE']
  flux_hdu      = template_hdu['FLUX']
  physical_hdu  = template_hdu['PHYSICAL']
  detection_hdu  = template_hdu['DETECTION']
  photometry_hdu  = template_hdu['PHOTOMETRY']

  #create the separate catalogs
  position_cat = fits.FITS_rec.from_columns(position_hdu.columns, nrows=n, fill=True)
  shape_cat    = fits.FITS_rec.from_columns(shape_hdu.columns,    nrows=n, fill=True)
  flux_cat     = fits.FITS_rec.from_columns(flux_hdu.columns,     nrows=n, fill=True)
  physical_cat = fits.FITS_rec.from_columns(physical_hdu.columns, nrows=n, fill=True)
  detection_cat = fits.FITS_rec.from_columns(detection_hdu.columns, nrows=n, fill=True)
  photometry_cat = fits.FITS_rec.from_columns(photometry_hdu.columns, nrows=n, fill=True)

  position_cat['id'][:nxi]     = idx[xi]
  position_cat['x_tile'][:nxi] = np.zeros_like(q[xi]) #pixel location
  position_cat['y_tile'][:nxi] = np.zeros_like(q[xi]) #pixel location
  shape_cat['rhalf'][:nxi]     = rhalf[xi] #rhalf in arcsec
  shape_cat['sersic'][:nxi]    = sersic[xi]
  shape_cat['q'][:nxi]         = q[xi]
  shape_cat['pa'][:nxi]        = pa[xi]

  print(len(xi),nxi)
  for band in flux_hdu.data.names:
    if(band!='id'):
      flux_cat[band][:nxi]    = flux[band][xi]

  #for band in physical_hdu.data.names:
  #  if(band!='id'):
  #    physical_cat[band][:nxi]    = physical[band][xi]

  #restrict to perfect square
  nsqrt = int(np.sqrt(nxi))
  nlim = int(nsqrt**2)

  print(f'n = {n} nlim = {nlim}')

  #distribute on a grid

  #grid size
  nx = header['NAXIS1']
  ny = header['NAXIS2']

  #area per object
  dnx = int(nx/float(nsqrt))
  if(dnx%2==0):
    dnx += 1
  dny = int(ny/float(nsqrt))
  if(dny%2==0):
    dny += 1

  #create permutation of indices
  irand = rng.permutation(nlim)

  #place on grid
  for i in range(nsqrt): 
    for j in range(nsqrt): 
      position_cat['x_tile'][irand[i*nsqrt+j]] = 0.5*dnx + i*dnx
      position_cat['y_tile'][irand[i*nsqrt+j]] = 0.5*dny + j*dny

  #remake hdu list

  #position_cat = fits.FITS_rec.from_columns(position_hdu.columns, nrows=n, fill=True)
  #shape_cat    = fits.FITS_rec.from_columns(shape_hdu.columns,    nrows=n, fill=True)
  #flux_cat     = fits.FITS_rec.from_columns(flux_hdu.columns,     nrows=n, fill=True)
  #physical_cat = fits.FITS_rec.from_columns(physical_hdu.columns, nrows=n, fill=True)
  #detection_cat = fits.FITS_rec.from_columns(detection_hdu.columns, nrows=n, fill=True)
  print(len(template_hdu['POSITION'].data.names),template_hdu['POSITION'].data.names,len(template_hdu['POSITION'].data['x_tile']))
  template_hdu['POSITION'].data = position_cat[:nlim]
  template_hdu['SHAPE'].data = shape_cat[:nlim]
  template_hdu['FLUX'].data     = flux_cat[:nlim]
  template_hdu['PHYSICAL'].data = physical_cat[:nlim]
  template_hdu['DETECTION'].data = detection_cat[:nlim]
  template_hdu['PHOTOMETRY'].data = photometry_cat[:nlim]
  #for name in  template_hdu['POSITION'].data.names:
  #  template_hdu['POSITION'].data[name][:nlim] = position_cat[name][xi[:nlim]]
  #template_hdu['SHAPE'].data[:nlim]    = shape_cat[xi[:nlim]]
  #template_hdu['FLUX'].data[:nlim]     = flux_cat[xi[:nlim]]
  #template_hdu['PHYSICAL'].data[:nlim] = physical_cat[xi[:nlim]]

  #primary = fits.PrimaryHDU(header=hdr)
  #with fits.HDUList([primary]) as hdul:
  #  hdul.append(position_cat,header=position_hdu.header,name=position_hdu.name)
  #  hdul.append(shape_cat)
  #  hdul.append(flux_cat)
  #  hdul.append(physical_cat)
  #  hdul.append(detection_cat)

  print(f"min sersic {np.min(sersic)}")
  print(f"max sersic {np.max(sersic)}")

  #return catalog objects in grid
  return nlim, [dnx,dny], template_hdu


'''
    # make image
    print(f"BAND = {args.band}")
    im = make_image(cat, n_pix_per_side_withbound, n_pix_per_gal,
        band=conf.frames[0]['band'],
        pixel_scale=conf.frames[0]['scale'],
        psf_in=psf_in, local_wcs=local_wcs)

    print(repr(header))


#########################################
# Read configuration file.
#########################################

def read_config(config_file):
    #Read configuration file.
    with open(config_file, "r") as c:
        conf = yaml.load(c, Loader=yaml.FullLoader)
    config = argparse.Namespace(**conf)
    return config
'''


#########################################
# Pull random numbers from a truncated gaussian
# using scipy stats
#########################################
def make_truncnorm(min=0, max=1, mu=0, sigma=1, **extras):
    a = (min - mu) / sigma
    b = (max - mu) / sigma
    return stats.truncnorm(a, b, loc=mu, scale=sigma)


#########################################
# Pull random numbers from a truncated gaussian
# using numpy random
#########################################
def make_truncnorm_np_random(rs, xdict, n=1):

    min = 0.0
    max = 1.0
    mu = 0.0
    sigma = 1.0

    if('min' in xdict):
        min = xdict['min']
    if('max' in xdict):
        max = xdict['max']
    if('mu' in xdict):
        mu = xdict['mu']
    if('sigma' in xdict):
        sigma = xdict['sigma']

#    a = (min - mu) / sigma
#    b = (max - mu) / sigma

    #print(f'config.sources["number"] {}')
    #exit()
    x = np.full(n,-1e9,dtype=np.float32)
    flag_iter = True
    while(flag_iter):
        idx = np.where( (x<min)|(x>max) )[0]
        if(len(idx)==0):
            flag_iter = False
        else:
            x_draw = rs.normal(loc=mu,scale=sigma,size=len(idx))
            x[idx] = x_draw
    return x



#########################################
# Create a source catalog
#########################################
def make_catalog(grid_points, n_gal, band="fclear", pixel_scale=0.03, noise=1.0, rs=None):
    '''
    Generate catalog.
    '''
    grid_keys = list(grid_points.keys())
    cols = np.unique(grid_keys + ["pa", "ra", "dec", "id", "x", "y", "n_pix","noise"] + [band])
    cat_dtype = np.dtype([(c, np.float64) for c in cols])
    cat = np.zeros(n_gal, dtype=cat_dtype)
    for i, k in enumerate(grid_keys):
        cat[k] = grid_points[k]
    n_pix = np.pi * (cat["rhalf"] / pixel_scale)**2
    cat[band] = 2 * cat["snr"] * np.sqrt(n_pix) * noise
    cat["n_pix"] = n_pix
    if(rs is None):
        cat["pa"] = np.random.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=(n_gal))
    else:
        cat["pa"] = rs.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=(n_gal))
    return cat


#########################################
# Generate image with galfit.
#########################################
def make_image(cat, header, n_pix_per_gal, band="F200W", pixel_scale=0.03, psf=None, wcs=None, origin=0): 
               
    '''
    Generate image with galfit.
    '''

    image = galsim.ImageF(header['NAXIS1'], header['NAXIS2'], scale=pixel_scale)

    #if(psf_in is None):
    #    print("Using gaussian psf....")
    #    psf = galsim.Gaussian(flux=1., sigma=sigma_psf)
    #else:
    #print("Using custom psf...")
    local_wcs = galsim.PixelScale(pixel_scale).withOrigin(image.center)
    psf = galsim.InterpolatedImage(galsim.Image(psf), wcs=local_wcs)

    print("Beginning image convolution....")
    time_start = time.time()

    gsp = galsim.GSParams(maximum_fft_size=10240)

    #for i, row in enumerate(cat):

    dx_off = 0#0.5*(n_pix_per_gal[0]-1)
    dy_off = 0#0.5*(n_pix_per_gal[1]-1)
    for i in range(len(cat['POSITION'].data['x_tile'])):
        pos   = cat['POSITION'].data
        shape = cat['SHAPE'].data
        flux  = cat['FLUX'].data
        #print(f"Object {i} flux {row['flux']}")
        #print(f"Object {i} flux {row['flux'][i]}")
        if(flux[band][i]==0):
          print(f'flux is zero in this band for {i}')
          print(f'flux array {flux[i]}')
          #exit()
        gal = galsim.Sersic(half_light_radius=shape["rhalf"][i],
                            n=shape["sersic"][i], flux=flux[band][i])
        egal = gal.shear(q=shape["q"][i], beta=shape["pa"][i] * galsim.degrees )
        final_gal = galsim.Convolve([psf, egal], gsparams=gsp) #skip convolution
        #final_gal = egal

        # place the galaxy and draw it
        x, y = pos["x_tile"][i] + 1, pos["y_tile"][i] + 1


      
        pix_bounds = np.array([dx_off+ (x - 0.5*(n_pix_per_gal[0]-1) + 1), dx_off + (x + 0.5*(n_pix_per_gal[0]-1) - 1),
                      dy_off+ (y - 0.5*(n_pix_per_gal[1]-1) + 1), dy_off + (y + 0.5*(n_pix_per_gal[1]-1) - 1)]).astype(int)

        #print(f'gal {i} pixel_bounds {pix_bounds}')
        bounds = galsim.BoundsI(pix_bounds[0],pix_bounds[1],pix_bounds[2],pix_bounds[3])
                                
      
        final_gal.drawImage(image[bounds], add_to_image=True)
    time_end = time.time()

    print(f"Time to perform image convolution {time_end-time_start}s.")

    return image


#########################################
# Make the output FITS header
#########################################
def make_header(config, idx_band):
    '''
    Generate header.
    '''
    pixel_scale = config.frames[idx_band]['scale']
    header = {}
    header["CRPIX1"] = 0.0
    header["CRPIX2"] = 0.0
    header["CRVAL1"] = config.ra
    header["CRVAL2"] = config.dec
    header["CD1_1"] = -config.frames[idx_band]['scale'] / 3600.
    header["CD2_2"] = config.frames[idx_band]['scale'] / 3600.
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["FILTER"] = config.frames[idx_band]['band']
    header["NOISE"] = config.frames[idx_band]['noise']
    header["PSFSIGMA"] = config.frames[idx_band]['fwhm']
    header["PIXSCALE"] = config.frames[idx_band]['scale']
    wcs = WCS(header)
    return header, wcs

#########################################
# Write the output fits image
#########################################
def write_image(filename, header, noiseless, cat, **kwargs):

    '''
    Write image to fits file.
    '''

    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr.update(header)
    hdr.update(EXT1="SCI", EXT2="POSITION", EXT3='SHAPE', EXT4='FLUX', EXT5='PHYSICAL')
    hdr.update(**kwargs)
    primary = fits.PrimaryHDU(header=hdr)
    with fits.HDUList([primary]) as hdul:
        hdul.append(fits.ImageHDU(noiseless, header=hdr,name='SCI'))
        for cat_hdu in cat[1:]:
          hdul.append(cat_hdu)
          print(cat_hdu.data.names)
        hdul.writeto(filename, overwrite=True)


#########################################
# Load a PSF image
#########################################
def load_psf(args):

    fname = args.psf

    psf = fits.getdata(fname)

    local_wcs = WCS(fits.getheader(fname))

    return psf, local_wcs


######################################
# main function
######################################
def main():

    #F444W FWHM
    #https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
    fwhm_f444w = 0.145 #arcsec

    #make the parser
    parser = create_parser()

    # read args
    args = parser.parse_args()

    #if(args.verbose):
    print(f'Input catalog {args.cat}')
    print(f'Image header  {args.header}')
    print(f'Output image  {args.output}')
    print(f'Band          {args.band}')
    print(f'Image origin  {args.origin}')
    print(f'RNG seed      {args.seed}')
    print(f'ID multiple   {args.id_fac}')
    print(f'PSF           {args.psf}')
    print(f'Data model    {args.data_model}')

    #start timer
    time_begin = time.time()

    #read header
    header = fits.Header.fromfile(args.header, sep='\n', endcard=False, padding=False)

    #get wcs
    wcs = WCS(header)

    #read psf
    psf = fits.getdata(args.psf)

    #read catalog
    input_catalog_hdu = fits.open(args.cat)

    #get pixel scale
    pixel_area = np.abs(np.linalg.det(wcs.pixel_scale_matrix*3600))
    pixel_scale = pixel_area**0.5
    print(f'Pixel area  = {pixel_area} arcsec^2.')
    print(f'Pixel scale = {pixel_scale} arcsec.')

    #data model:
    # hdu['position'].data.names
    # ['ID','ra','dec']
    # hdu['shape'].data.names
    # ['pa','q','rhalf','sersic']
    # hdu['flux'].data.names
    # ['F070W','F090W','F115W','F150W','F200W','F277W','F335M','F356W','F410M','F444W']

    rhalf = input_catalog_hdu['shape'].data['rhalf']
    #assume rhalf in pixels, convert to arcsec
    rhalf *= pixel_scale
    rhalf = rhalf**2 - (0.5*fwhm_f444w)**2
    rhalf[rhalf<(0.5*fwhm_f444w)**2] = (0.5*fwhm_f444w)**2
    rhalf = rhalf**0.5

    print(f'0.5 FWHM of F444W in arcsec {0.5*fwhm_f444w}')

    pa    = input_catalog_hdu['shape'].data['pa']
    q     = input_catalog_hdu['shape'].data['q']
    print(f'rhalf info min {np.nanmin(rhalf)} mean {np.nanmean(rhalf)} median {np.nanmedian(rhalf)} std {np.nanstd(rhalf)} max {np.nanmax(rhalf)}')
    print(f'pa info min {np.nanmin(pa)} mean {np.nanmean(pa)} median {np.nanmedian(pa)} std {np.nanstd(pa)} max {np.nanmax(pa)}')
    print(f'q info min {np.nanmin(q)} mean {np.nanmean(q)} median {np.nanmedian(q)} std {np.nanstd(q)} max {np.nanmax(q)}')

    #build image catalog from input catalog model
    n, n_pix_per_gal, catalog = refine_catalog(input_catalog_hdu, header, seed=args.seed, data_model=args.data_model)

    print(f"Total number of objects remaining in refined catalog {n}.")
    print(f"Length of refined catalog {len(catalog['FLUX'].data)}.")


    # make image
    print(type(catalog))
    im = make_image(catalog, header, n_pix_per_gal, band=args.band,
        pixel_scale=pixel_scale,
        psf=psf, wcs=wcs)

    #generate noisless image
    #noiseless = im.copy().array[int(n_pix_per_gal):-int(n_pix_per_gal), int(n_pix_per_gal):-int(n_pix_per_gal)]
    noiseless = im.copy().array[:,:] #bare image

    #check total flux
    total_flux = np.sum(noiseless)
    total_obj_flux = np.sum(catalog['flux'].data[args.band])
    print(f'Sum of object fluxes {total_obj_flux} total image flux {total_flux} ratio {(total_flux-total_obj_flux)/total_obj_flux}')
    print(f"Statistical properties of flux min {np.min(catalog['flux'].data[args.band])} std {np.nanstd(catalog['flux'].data[args.band])} mean {np.mean(catalog['flux'].data[args.band])} median {np.median(catalog['flux'].data[args.band])} max {np.max(catalog['flux'].data[args.band])}")

    #conversion between image units and nJy
    flux_to_nJy  = 1e15 / 4.25e10 * pixel_area #header_flux['PIXAR_SR']*1.e15 # to nJy

    #convert FROM nJy to image units
    print(f'Converting from nJy to image units, dividing by {flux_to_nJy}.')
    noiseless/=flux_to_nJy

    # generate header and get WCS
    ra, dec = wcs.all_pix2world(catalog['POSITION'].data["x_tile"], catalog['POSITION'].data["y_tile"], args.origin)
    catalog['POSITION'].data["ra"] = ra
    catalog['POSITION'].data["dec"] = dec

     
    for i,t in enumerate(catalog[1:]):
      if(i==0):
        ids = catalog[t].data['id'].copy()
      catalog[t].data['id'] = ids + args.seed*args.id_fac

    print(f"Min/max of catalog ids {np.min(catalog[1].data['id'])}/{np.max(catalog[1].data['id'])}")

    # write image
    write_image(args.output, header, noiseless, catalog)
    print(f"Successfully wrote image and catalog to {args.output}.")


    time_end = time.time()
    print(f'Total time to create injection image = {time_end-time_begin}s.')

# run the program
if __name__ == "__main__":
    main()
