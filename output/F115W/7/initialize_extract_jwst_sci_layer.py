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
#FDIR_EL    = '/home/brant/github/jades-pipeline/detection/extract_layers/'
#FDIR_REFIMG  = '/data/groups/comp-astro/jades/jades-data/GOODS-S/images/JWST/JADES-Deep-Public/v0.8'
#FDIR_DATA   = f'{FDIR_JADES}/jades-data/GOODS-S/images/JWST/JADES/v0.8/'
#FDIR_PSFS    = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/psfs'
#CATALOG = '/data/groups/comp-astro/brant/source_injection/automation/mock_injection_cat.fits'
#FDIR_MNI = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/make_noise_image/noise'

#########################################
## Create a parser for the command line
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Create a simulated image with sources for injection into JADES mosaics.")

    parser.add_argument("--input",
        type=str,
        default='input.fits',
        help='Input image to extract sci layer from.')

    parser.add_argument("--output",
        type=str,
        default='output_sci.fits',
        help='Output sci layer image.')

    return parser

####################
## generate images
####################
def create_extract_layer(args):

  #create directory and enter it
  subroutine = 'extract_layers'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)

  #link the insertion
  fdir_image = 'insert_images'
  try:
    os.symlink(f'{FDIR_EXEC}/{fdir_image}',fdir_image)
  except:
    pass



  #link the script
  fsubr = f'extract_jwst_sci_layer.py'
  fname = f'{FDIR_EL}/{fsubr}'
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
  command = f'python3 {fsubr} {args.input} {args.output}\n'
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

  #create extract_layer
  create_extract_layer(args)

# run the script
if __name__=="__main__":
  main()
