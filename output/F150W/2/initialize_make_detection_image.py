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
#FDIR_REFIMG  = '/data/groups/comp-astro/jades/jades-data/GOODS-S/images/JWST/JADES-Deep-Public/v0.8'
#FDIR_DATA   = f'{FDIR_JADES}/jades-data/GOODS-S/images/JWST/JADES/v0.8/'
#FDIR_PSFS    = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/psfs'
#CATALOG = '/data/groups/comp-astro/brant/source_injection/automation/mock_injection_cat.fits'
#
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
def create_make_detection_image(args):

  #create directory and enter it
  subroutine = 'make_detection_image'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)

  #make the output directory
  #fdir = 'stacks'
  #try:
  #  os.mkdir(fdir)
  #except:
  #  pass

  #link the generate images directory
  fdir = f'{args.fdir_insert_images}'
  try:
    os.symlink(fdir,'insert_images')
  except:
    pass
  #link the script
  fsubr = f'add_injection_catalog_to_image.py'
  fname = f'{FDIR_SI}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass

  #link the script
  fsubr = f'make_detection_image.py'
  fname = f'{FDIR_MDI}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass




  nbands = len(args.bands)

  #open the script to run the python script
  fp = open(f'filter_list.txt','w')
  for iband, band in enumerate(args.bands):
    command = f'insert_images/stacks/{band.lower()}.injected.fits\n'
    fp.write(command)
  fp.close()

  fp = open(f'err_list.txt','w')
  for iband, band in enumerate(args.bands):
    command = f'{FDIR_MNI}/{band.lower()}_MED_NOISE.fits\n'
    fp.write(command)
  fp.close()

  #open the script to run the python script
  fp = open(f'run_{subroutine}.sh','w')


  #write the command
  #counter = 0
  #for iband, band in enumerate(args.bands):

    #open the list of output files
  command = f'python3 {fsubr} -i filter_list.txt -e err_list.txt -o snr.injected.fits -iv -v\n'
  fp.write(command)
  command = f'python3 add_injection_catalog_to_image.py -i snr.injected.fits -c insert_images/stacks/{args.bands[0].lower()}.injected.fits -o snr.injected.fits\n'
  fp.write(command)
  command = f'python3 add_injection_catalog_to_image.py -i snr.injected.fits.signal -c insert_images/stacks/{args.bands[0].lower()}.injected.fits -o snr.injected.fits.signal\n'
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

  #create make_detection_image
  create_make_detection_image(args)

# run the script
if __name__=="__main__":
  main()
