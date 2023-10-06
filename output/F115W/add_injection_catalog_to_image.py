import os
import sys
import numpy as np
import argparse
from astropy.io import fits

FDIR_EXEC  = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2-completeness/F115W/'
FDIR_TILES = '/home/brant/github/jades-pipeline/detection/source_injection'
FDIR_JADES = '/data/groups/comp-astro/jades'
FIELD      = 'v0.8_gs'
VERSION    = 'v0.8.2'
FDIR_OBS   = f'{FDIR_JADES}/{FIELD}/{VERSION}'
FDIR_SI    = '/home/brant/github/jades-pipeline/detection/source_injection'
FDIR_REFIMG  = '/data/groups/comp-astro/jades/jades-data/GOODS-S/images/JWST/JADES/v0.8'
FDIR_PSFS    = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/psfs'
CATALOG = f'{FDIR_EXEC}/injection-input_F115W-dropouts.fits'

#########################################
## Create a parser for the command line
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Create a simulated image with sources for injection into JADES mosaics.")

    parser.add_argument("-i","--input",
        type=str,
        default='input.fits',
        help='Input image to add catalog to.')

    parser.add_argument("-c","--cat",
        type=str,
        default='image_with_catalog.fits',
        help='Input image with catalog to propagate.')

    parser.add_argument("-o","--output",
        type=str,
        default='output.fits',
        help='Output image to write with catalog.')

    return parser

####################
## main function
####################
def main():

  #make the parser
  parser = create_parser()

  # read args
  args = parser.parse_args()

  hdu_in  = fits.open(args.input)
  hdu_cat = fits.open(args.cat)


  table_list = ['POSITION','SHAPE','FLUX','PHYSICAL','DETECTION','PHOTOMETRY'] 
  for table in table_list:
    print(f'Appending {table}...')
    hdu_in.append(hdu_cat[table])

  hdu_in.writeto(args.output,overwrite=True)

# run the script
if __name__=="__main__":
  main()
