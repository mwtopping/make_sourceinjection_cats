import os
import sys
import numpy as np
import argparse
from jades_completeness_defs import *

#FDIR_EXEC  = '/data/groups/comp-astro/brant/source_injection/automation'
#FDIR_TILES = '/home/brant/github/jades-pipeline/detection/source_injection'
#FDIR_JADES = '/data/groups/comp-astro/jades'
#FIELD      = 'v0.8_gs'
#VERSION    = 'v0.8.2'
#FDIR_OBS   = f'{FDIR_JADES}/{FIELD}/{VERSION}'
#FDIR_SI    = '/home/brant/github/jades-pipeline/detection/source_injection'
#FDIR_MDI    = '/home/brant/github/jades-pipeline/detection/make_detection_image'
#FDIR_DET    = '/home/brant/github/jades-pipeline/detection/detection'
#FDIR_DETROOT    = '/home/brant/github/jades-pipeline/detection'
#FDIR_RES    = '/home/brant/github/jades-pipeline/detection/image_rescaling'
#FDIR_REFIMG  = '/data/groups/comp-astro/jades/jades-data/GOODS-S/images/JWST/JADES-Deep-Public/v0.8'
#FDIR_DATA   = f'{FDIR_JADES}/jades-data/GOODS-S/images/JWST/JADES/v0.8/'
#FDIR_PSFS    = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/psfs'
#FDIR_SSEG   = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/make_star_segmentation'
#CATALOG = '/data/groups/comp-astro/brant/source_injection/automation/mock_injection_cat.fits'
#FDIR_MNI = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/make_noise_image/noise'

#########################################
## Create a parser for the command line
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Create a simulated image with sources for injection into JADES mosaics.")

    parser.add_argument("--bands",
        nargs="+",
        type=str,
        default=['F444W'],
        help='Bands to create simulated images.')

    parser.add_argument("--fdir_insert_images",
        type=str,
        default='insert_images',
        help='Directory insert_images.')

    return parser

####################
## generate images
####################
def create_detection(args):

  #create directory and enter it
  subroutine = 'detection'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)


  #link the make_detection_image
  fdir = 'make_detection_image'
  try:
    os.symlink(f'{FDIR_EXEC}/{fdir}',fdir)
  except:
    pass

  fdir = 'insert_images'
  try:
    os.symlink(f'{FDIR_EXEC}/{fdir}',fdir)
  except:
    pass

  #link the script
  fsubr = 'jades_detection_rc10.py'
  fname = f'{FDIR_DET}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass

  #link the script
  fsubr = 'detect_and_mask.py'
  fname = f'{FDIR_DET}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass


  #link the script
  fsubr = 'jades_photutils_interface.py'
  fname = f'{FDIR_DETROOT}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass


  #link the script
  fsubr = 'make_colorized_segmap.py'
  fname = f'{FDIR_RES}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass

  #link the script
  fsubr = 'jades_basic_detection_catalog.py'
  fname = f'{FDIR_DET}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass


  #open the script to run the python script
  fp = open(f'run_{subroutine}.sh','w')


  #write the command
  #counter = 0
  #for iband, band in enumerate(args.bands):

  #open the list of output files
  command = f'python3 jades_detection_rc10.py -v --snr make_detection_image/snr.injected.fits --output det.fits -t 1.5 --npix 1 --di insert_images/stacks/f200w.injected.sci.fits --deblend --centroid centroid --gpu --star_segmap {FDIR_SSEG}/star_only_segmap.fits\n'
  fp.write(command)
  command = f'python3 make_colorized_segmap.py segmap.fits segmap.png\n'
  fp.write(command)
  command = f'python3 jades_basic_detection_catalog.py -v -f make_detection_image/snr.injected.fits.signal -s segmap.fits --output det.cat --aperture_mask --limit_kron --fix_nan_pos\n'
  fp.write(command)
  fp.close()

#

####################
## main function
####################
def main():

  #make the parser
  parser = create_parser()

  # read args
  args = parser.parse_args()

  #create detection
  create_detection(args)

# run the script
if __name__=="__main__":
  main()
