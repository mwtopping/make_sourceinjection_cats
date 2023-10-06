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
#FDIR_REFIMG  = '/data/groups/comp-astro/jades/jades-data/GOODS-S/images/JWST/JADES-Deep-Public/v0.8'
#FDIR_DATA   = f'{FDIR_JADES}/jades-data/GOODS-S/images/JWST/JADES/v0.8/'
#FDIR_PSFS    = '/data/groups/comp-astro/jades/v0.8_gs/v0.8.2/psfs'
#CATALOG = '/data/groups/comp-astro/brant/source_injection/automation/mock_injection_cat.fits'

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

    parser.add_argument("--fdir_embed_subimages",
        type=str,
        default='embed_subimages',
        help='Directory embed_subimages.')

    parser.add_argument("--blank",
        type=str,
        default=None,
        help='Blank image for inserting test objects.')

    return parser

####################
## generate images
####################
def create_insert_images(args):

  #create directory and enter it
  subroutine = 'insert_images'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)

  #make the output directory
  fdir = 'stacks'
  try:
    os.mkdir(fdir)
  except:
    pass

  #link the generate images directory
  fdir = f'{args.fdir_embed_subimages}'
  try:
    os.symlink(fdir,'embed_subimages')
  except:
    pass

  #link the script
  fsubr = f'jades_insert_images.py'
  fname = f'{FDIR_SI}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass

  #open the script to run the python script
  fp = open(f'run_{subroutine}.sh','w')


  #write the command
  nbands = len(args.bands)
  counter = 0
  for iband, band in enumerate(args.bands):

    #open the list of output files
    command = f'python3 {fsubr} --fdir_embed_subimages {args.fdir_embed_subimages} --flist_embedded embedded_list.{band.upper()}.txt --input {FDIR_DATA}/{band.upper()}/mosaic_{band.upper()}.fits --output stacks/{band.lower()}.injected.fits -v\n'
    fp.write(command)

  #also make blank images?
  if(args.blank is not None):
    for iband, band in enumerate(args.bands):
      #open the list of output files
      command = f'python3 {fsubr} --fdir_embed_subimages {args.fdir_embed_subimages} --flist_embedded embedded_list.{band.upper()}.txt --input {args.blank} --output blank.{band.lower()}.injected.fits -v\n'
      fp.write(command)

  fp.close()
  

#headers
#python3 jades_generate_images.py --cat mock_injection_cat.fits  --psf mpsf_f200w.fits --header headers/header_0.txt --output output/injection_image.0.
#fits --seed 0

####################
## main function
####################
def main():

  #make the parser
  parser = create_parser()

  # read args
  args = parser.parse_args()

  #create insert_images
  create_insert_images(args)

# run the script
if __name__=="__main__":
  main()
