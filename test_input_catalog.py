from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

hdu = fits.open('injection-input_F150W-dropouts_test.fits')

def abmag(flux):
    return -2.5*np.log10(flux*1e-9/3631)

for x in hdu:
    print(x.name)
    try:
        print(hdu[x].columns)
    except:
        pass

lambdas = [.704, .901, 1.154, 1.501, 1.990, 2.786, 3.365, 3.563, 4.092, 4.442] 

fig, ax = plt.subplots()

for row in tqdm(hdu['flux'].data):
    mags = []
    for f in ['F070W', 'F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F335M', 'F356W', 'F410M', 'F444W']:
        mags.append(abmag(row[f]))


    ax.plot(lambdas, mags, alpha=0.1)

ax.set_ylim([35, 20])
plt.show()

























