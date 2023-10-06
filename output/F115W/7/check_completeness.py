from astropy.io import fits
import numpy as np
import sys

fname = 'completeness.fits'
if(len(sys.argv)>1):
  fname = sys.argv[1]
hdu = fits.open(fname)
det  = hdu['DETECTION'].data['detected']
xi = np.where(det==True)[0]
print(f'Fraction detected {float(len(xi))/float(len(det))}.')

flux = hdu['FLUX'].data['F444W']
fi = np.argsort(-1.0*flux)
nobj = len(flux)
comp = np.zeros_like(flux)
for i in range(len(comp)):
    if(i>0):
        comp[i] = comp[i-1]
    if(det[fi[i]]):
        comp[i] += 1.0
comp_poss = np.arange(nobj)+1
comp/=comp_poss
mab = 31.4-2.5*np.log10(flux[fi])
xi = np.where(comp>=0.995)
xcc = np.max(mab[xi])
print(f'99.5% Completeness at {xcc:3.2f}')
xi = np.where(comp>=0.99)
xcc = np.max(mab[xi])
print(f'99.0% Completeness at {xcc:3.2f}')
xi = np.where(comp>=0.95)
xcc = np.max(mab[xi])
print(f'95.0% Completeness at {xcc:3.2f}')
xi = np.where(comp>=0.9)
xcc = np.max(mab[xi])
print(f'90.0% Completeness at {xcc:3.2f}')
