from astropy.io import fits
import numpy as np
hdu = fits.open('f200w.fits')

hdu['SCI'].data = np.zeros_like(hdu['SCI'].data,dtype=np.float32)

hdu.writeto('blank.fits',overwrite=True)
