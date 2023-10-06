from astropy.io import fits
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



#hdu = fits.open('./F150W/f150w_completeness.fits')
hdu = fits.open('./F115W/f115w_completeness.fits')

zhdu = fits.open('../injection-input_F115W-dropouts_withz.fits')


for x in hdu:
    print(x.name)
    try:
        print(x.columns)
    except:
        pass


#for row in hdu['position'].data:
#    print(row)

for ii, n in tqdm(enumerate(hdu['shape'].data['sersic']), total=len(hdu['shape'].data['id'])):
    ind = np.where(zhdu['shape'].data['sersic'] == n)[0][0]
    z = zhdu['physical'].data['redshift'][ind]
    hdu['physical'].data['redshift'][ii] = z



print(hdu['physical'].data)

hdu.writeto('f115w_completeness-withz.fits')
#arr = hdu['position'].data['id']
#print(arr)
#plt.hist(arr)
#plt.show()
