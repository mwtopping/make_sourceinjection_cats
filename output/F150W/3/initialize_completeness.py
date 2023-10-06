import os
import sys
import numpy as np
import argparse
from jades_completeness_defs import *

#FDIR_EXEC    = '/data/groups/comp-astro/brant/source_injection/automation'
#FDIR_DETROOT = '/home/brant/github/jades-pipeline/detection'
#CATALOG      = f'{FDIR_EXEC}/mock_injection_cat.fits'
#FDIR_SI      = f'{FDIR_DETROOT}/source_injection'
#FDIR_TILES   = f'{FDIR_SI}'
#FDIR_JADES   = '/data/groups/comp-astro/jades'
#FIELD        = 'v0.8_gs'
#VERSION      = 'v0.8.2'
#FDIR_OBS     = f'{FDIR_JADES}/{FIELD}/{VERSION}'
#FDIR_MDI     = f'{FDIR_DETROOT}/make_detection_image'
#FDIR_DET     = f'{FDIR_DETROOT}/detection'
#FDIR_RES     = f'{FDIR_DETROOT}/image_rescaling'
#FDIR_REFIMG  = f'{FDIR_JADES}/jades-data/GOODS-S/images/JWST/JADES-Deep-Public/v0.8' 
#FDIR_DATA    = f'{FDIR_JADES}/jades-data/GOODS-S/images/JWST/JADES/v0.8/'
#FDIR_PSFS    = f'{FDIR_OBS}/psfs'
#FDIR_SSEG    = f'{FDIR_OBS}/make_star_segmentation'
#FDIR_MNI     = f'{FDIR_OBS}/make_noise_image/noise'

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
def create_completeness(args):

  #create directory and enter it
  subroutine = 'completeness'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)


  #link the script
  fsubr = 'jades_completeness.py'
  fname = f'{FDIR_SI}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass

  fsubr = 'check_completeness.py'
  fname = f'{FDIR_SI}/{fsubr}'
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

  #open the script to run the python script
  fp = open(f'run_{subroutine}.sh','w')


  #write the command
  #counter = 0
  #for iband, band in enumerate(args.bands):

  #open the list of output files
#python3 jades_completeness.py --input /data/groups/comp-astro/brant/source_injection/automation/make_detection_image/snr.injected.fits --cat /data/groups/comp-astro/brant/source_injection/automation/detection/det.cat --segmap /data/groups/comp-astro/brant/source_injection/automation/detection/segmap.fits --output completeness.fits
  command = f'python3 jades_completeness.py --input {FDIR_EXEC}/make_detection_image/snr.injected.fits  --cat {FDIR_EXEC}/detection/det.cat --segmap {FDIR_EXEC}/detection/segmap.fits --output completeness.fits\n'
  fp.write(command)

  command = f'python3 check_completeness.py\n'
  fp.write(command)
  #command = f'python3 make_colorized_segmap.py segmap.fits segmap.png\n'
  #fp.write(command)



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

  #create completeness
  create_completeness(args)

# run the script
if __name__=="__main__":
  main()
