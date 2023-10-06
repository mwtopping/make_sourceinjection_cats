from astropy.io import fits
import numpy as np
import sys

fname = 'f200w.fits'
if(len(sys.argv)>1):
  fname = sys.argv[1]
hdu = fits.open(fname)

hdu['SCI'].data = np.zeros_like(hdu['SCI'].data,dtype=np.float32)

hdu.writeto('blank.fits',overwrite=True)
