import os
import sys
import numpy as np
import argparse
from jades_completeness_defs import *

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

    parser.add_argument("--cat",
        type=str,
        default='input_catalog.fits',
        help='Input catalog detailing the injected source population.')

    parser.add_argument("--data_model",
        type=str,
        default='injection+recovery_data_model.fits',
        help='Data model FITS for injection catalog.')

    parser.add_argument("--bands",
        nargs="+",
        type=str,
        default=['F444W'],
        help='Bands to create simulated images.')

    return parser

####################
## generate images
####################
#def create_generate_images(args,bands=['f444w']):
def create_generate_images(args):

  #create directory and enter it
  subroutine = 'generate_images'
  fdir = f'{subroutine}'
  try:
    os.mkdir(fdir)
  except:
    pass
  os.chdir(fdir)

  #make the output directory
  fdir = 'output'
  try:
    os.mkdir(fdir)
  except:
    pass

  #link the psfs
  fdir = f'{FDIR_PSFS}'
  try:
    os.symlink(fdir,'psfs')
  except:
    pass

  #link the headers
  fdhead = 'generate_headers'
  try:
    os.symlink(f'../{fdhead}',fdhead)
  except:
    pass

  

  #link the script
  fsubr = f'jades_{subroutine}.py'
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
    fpoutput = open(f'injection_image_list.{band.lower()}.txt','w')
    fptile = open(f'{fdhead}/tiles.{band.lower()}.txt','r')
    tiles = fptile.readlines()
    tiles = [tile.strip('\n') for tile in tiles]
    fptile.close()
    fpseeds = open(f'{fdhead}/seeds.{band.lower()}.txt','r')
    seeds = np.asarray(fpseeds.readlines(),dtype=np.int32)
    fpseeds.close()

    for itile, tile in enumerate(tiles):
      output_name = f'injection_image.{band}.{seeds[itile]}.fits'
      fpoutput.write(output_name+'\n')
      command = f'python3 {fsubr} --cat {args.cat} --psf psfs/mpsf_{band.lower()}.fits --header {fdhead}/{tile} --output output/{output_name} --seed {seeds[itile]} --data_model {args.data_model} --band {band.upper()}\n'
      fp.write(command)
      counter+=1
    fpoutput.close()
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
  create_generate_images(args)

  #create embed_subimages
  #create_embed_subimages()

  #create insert_images
  #create_insert_images()

# run the script
if __name__=="__main__":
  main()
