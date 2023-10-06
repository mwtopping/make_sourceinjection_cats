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

    parser.add_argument("--fdir_generate_images",
        type=str,
        default='generate_images',
        help='Directory generate_images.')

    parser.add_argument("--full_header",
        type=str,
        default='f200w.fits',
        help='Image with the full header WCS.')

    return parser

####################
## generate images
####################
def create_embed_subimages(args):

  #create directory and enter it
  subroutine = 'embed_subimages'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)

  #make the output directory
  fdir = 'embed'
  try:
    os.mkdir(fdir)
  except:
    pass

  #link the generate images directory
  fdir = f'{args.fdir_generate_images}'
  try:
    os.symlink(fdir,'generate_images')
  except:
    pass

  #link the script
  fsubr = f'embed_injection_image.py'
  fname = f'{FDIR_SI}/{fsubr}'
  try:
    os.symlink(fname,fsubr)
  except:
    pass

  #open the script to run the python script
  fp = open(f'run_{subroutine}.sh','w')

  command = f'export FEXAM={args.full_header}\n'
  fp.write(command)



  #write the command
  nbands = len(args.bands)
  counter = 0
  for iband, band in enumerate(args.bands):

    #open the list of output files
    fname = f'{args.fdir_generate_images}/injection_image_list.{band.lower()}.txt'
    print(f'Attempting to open {fname}')

    fpinput = open(fname)
    inputs = fpinput.readlines()
    inputs = [finput.strip('\n') for finput in inputs]
    fpinput.close()


    for iinput, finput in enumerate(inputs):

      #define subimage
      command = f'export FSUB=generate_images/output/{finput}\n'
      fp.write(command)

      #define embedded name
      command = f'export FIMAG=embed/{finput.replace(".fits",".embedded.fits")}\n'
      fp.write(command)
      
      command = f'python3 {fsubr} --sub_image $FSUB --full_header $FEXAM --full_image $FIMAG\n'
      fp.write(command)

    
  for iband, band in enumerate(args.bands):
    command = f'ls embed/*{band.upper()}*.fits > embedded_list.{band.upper()}.txt\n'
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

  #bands = ['f277w','f335m','f356w','f410m','f430m','f444w','f460m','f480m']
  #bands = ['f444w','f460m']

  #create generate_images
  #create_generate_images(args,bands=bands)
  #create_generate_images(args)

  #create embed_subimages
  create_embed_subimages(args)

  #create insert_images
  #create_insert_images()

# run the script
if __name__=="__main__":
  main()
