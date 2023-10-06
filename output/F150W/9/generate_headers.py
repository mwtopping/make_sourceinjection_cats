
import os, argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname_image", type=str, default="mosaic.fits")
    parser.add_argument("--path_header", type=str, default=".")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--frac_data_accept", type=float, default=0.8)
    args = parser.parse_args()

    # make sure we use even numbers
    assert np.mod(args.image_size, 2) == 0

    # read in mosaic
    hdu_mosaic = fits.open(args.fname_image)
    mosaic_hdr = hdu_mosaic['SCI'].header
    mosaic_data = hdu_mosaic['ERR'].data


    band = fits.getheader(args.fname_image)['FILTER']
    print(f'Band = {band}')

    # setup pixel grid
    wcs_mosaic = WCS(mosaic_hdr)
    xshape, yshape = wcs_mosaic.pixel_shape
    xgrid = np.arange(0.5*args.image_size, xshape-0.5*args.image_size, args.image_size)
    ygrid = np.arange(0.5*args.image_size, yshape-0.5*args.image_size, args.image_size)
    xv, yv = np.meshgrid(xgrid, ygrid)

    fp = open(f'tiles.{band.lower()}.txt','w')
    fps = open(f'seeds.{band.lower()}.txt','w')
    # iterate over pixel grid, save only headers that contain data
    counter = 0
    for ii in range(xv.shape[0]):
        for jj in range(xv.shape[1]):
            co = Cutout2D(mosaic_data, (xv[ii][jj], yv[ii][jj]), args.image_size, wcs=wcs_mosaic)
            new_wcs = co.wcs
            frac_data = np.sum(co.data>0.0)/args.image_size**2
            if (frac_data > args.frac_data_accept):
                # save header wcs
                header = co.wcs.to_header()
                header.insert(1, ('NAXIS1', args.image_size, ''))
                header.insert(2, ('NAXIS2', args.image_size, ''))
                fname = f'header.{band.lower()}.{counter}.txt'
                header.tofile(os.path.join(args.path_header, fname), sep='\n', endcard=False, padding=False, overwrite=True)
                fp.write(f'{fname}\n')
                fps.write(f'{counter}\n')
            counter += 1
    fp.close()
    fps.close()
