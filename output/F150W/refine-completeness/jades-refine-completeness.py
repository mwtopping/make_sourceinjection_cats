import sys
import argparse
import time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from photutils.segmentation import SourceCatalog
from scipy.spatial import KDTree


#########################################
# Routine to parse command line arguments
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Detection flags and options from user.")


    parser.add_argument("--input", type=str,
                        default="input.fits")

    parser.add_argument("--output", type=str,
                        default="output.fits")

    parser.add_argument('--segmap',
                default='segmap.fits',
                type=str,
                help='Segmap with real objects identified.')

    parser.add_argument('--wht',
                default='wht.fits',
                type=str,
                help='Fits image with the weight layer.')

    parser.add_argument('-v', '--verbose',
                dest='verbose',
                action='store_true',
                help='Print helpful information to the screen? (default: False)',
                default=False)

    return parser

#########################################
# Main function
#########################################
def main():

    #make the parser
    parser = create_parser()

    # read args
    args = parser.parse_args()

    #input source catalog
    hdu = fits.open(args.input)  

    #segmap
    hdu_segmap = fits.open(args.segmap)  

    segmap = hdu_segmap['SCI'].data
  
    #wht
    hdu_wht = fits.open(args.wht)  

    print(hdu['POSITION'].data.names)

    #get a header
    wcs = WCS(hdu_wht['SCI'].header)

    #get positions
    ra  = hdu['POSITION'].data['ra']
    dec = hdu['POSITION'].data['dec']
    x,y = wcs.wcs_world2pix(ra,dec,0)

    data_wht = hdu_wht['WHT'].data
    data_texp = hdu_wht['EXP'].data

    #print(hdu['POSITION'].data['x_tile'][0:10])
    wht = np.zeros_like(x)
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    print(hdu['DETECTION'].data.names)
    print(f"Max wht { np.max(hdu['DETECTION'].data['wht']) }")
    print(f"Max texp { np.max(hdu['DETECTION'].data['texp']) }")
    hdu['DETECTION'].data['wht'] = data_wht[xi,yi]
    hdu['DETECTION'].data['texp'] = data_texp[xi,yi]
    print(f"After Max wht { np.max(hdu['DETECTION'].data['wht']) }")
    print(f"After Max texp { np.max(hdu['DETECTION'].data['texp']) }")

    detseg = segmap[xi,yi]
    #print(f"After Max detected { np.max(hdu['DETECTION'].data['detected']) }")

    hdu['DETECTION'].data['flag'][:] = 0
    print(f"Before Max flag { np.max(hdu['DETECTION'].data['flag'][:]) }")
    xi = np.where(detseg>0)

    print(len(xi[0]))

    hdu['DETECTION'].data['flag'][xi] = 1

    


    hdu.writeto(args.output,overwrite=True)
if __name__=="__main__":
  main()


