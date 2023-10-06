import os
import sys
import argparse
from jades_completeness_defs import *

#FDIR_JADES = '/data/groups/comp-astro/jades'
#FIELD      = 'v0.8_gs'
#VERSION    = 'v0.8.2'
#FDIR_OBS   = f'{FDIR_JADES}/{FIELD}/{VERSION}'
#FDIR_SI    = '/home/brant/github/jades-pipeline/detection/source_injection'
#FDIR_REFIMG  = '/data/groups/comp-astro/jades/jades-data/GOODS-S/images/JWST/JADES-Deep-Public/v0.8'

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

    return parser

####################
## generate headers
####################
def create_generate_headers(bands=['f200w'],image_size=2048):

  #create directory and enter it
  subroutine = 'generate_headers'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)

  #link the script
  fsubr = f'{subroutine}.py'
  fname = f'{FDIR_SI}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass

  #link the reference image
  for band in bands:
    reference_image = f'mosaic_{band.upper()}.fits'
    fname = f'{FDIR_REFIMG}/{band.upper()}/{reference_image}'
    try:
      os.symlink(fname,reference_image)
    except:
      pass

  #open the script to run the python script
  fp = open(f'run_{subroutine}.sh','w')

  #write the command
  for band in bands:
    reference_image = f'mosaic_{band.upper()}.fits'
    command = f'python3 {fsubr} --fname_image {reference_image} --image_size {image_size}\n'
    fp.write(command)
  fp.close()

  #back up
  os.chdir('..')

####################
## main function
####################
def main():

  #make the parser
  parser = create_parser()

  # read args
  args = parser.parse_args()

  #create generate_headers
  
  #bands = ['f277w','f335m','f356w','f410m','f430m','f444w','f460m','f480m']
  #bands = ['f277w','f335m','f356w','f410m','f444w']
  #bands = ['f444w','f460m']
  create_generate_headers(bands=args.bands)

# run the script
if __name__=="__main__":
  main()
