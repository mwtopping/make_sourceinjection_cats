import argparse
import numpy as np
import time as time
from astropy.io import fits

#########################################
# Routine to parse command line arguments
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Detection flags and options from user.")


    parser.add_argument('--fdir_embed_subimages',
        default='embed_subimages',
        type=str,
        help='Directory containing output from the embedded subimages.')

    parser.add_argument('--flist_embedded',
        default='embedded_list.txt',
        type=str,
        help='Filename containing list of injection images to insert.')

    parser.add_argument('--input',
        default='f200w.fits',
        metavar='input',
        type=str,
        help='Real image injection images are added to.')

    parser.add_argument('--output',
        default='f200w.with_injections.fits',
        metavar='output',
        type=str,
        help='Output images with injection images included.')

    parser.add_argument('-v', '--verbose',
                dest='verbose',
                action='store_true',
                help='Print helpful information to the screen? (default: False)',
                default=False)


    return parser



################################################
#main function
################################################
def main():


    #begin timer
    time_global_start = time.time()
    time_start = time.time()

    #create the command line argument parser
    parser = create_parser()

    #store the command line arguments
    args   = parser.parse_args()

    #open image
    print(f'Opening base image {args.input}.')
    hdu = fits.open(args.input)

    header = hdu['PRIMARY'].header

    band = header['FILTER'].strip(' ')

    print(f'Working with band {band}')

    #open list of injection images
    fplist = open(f'{args.fdir_embed_subimages}/{args.flist_embedded}')
    flist = fplist.readlines()
    flist = [fname.strip('\n') for fname in flist]
    fplist.close()


    #add projected images to foreground
    nrows = 0
    for i,fin in enumerate(flist):

      #open injected image
      fname = f'{args.fdir_embed_subimages}/{fin}'
      print(f'Injecting image {fin}.')
      hdu_data_injection = fits.open(fname)
      data_injection = hdu_data_injection['SCI'].data

      hdu['SCI'].data += data_injection




      nrows += hdu_data_injection['POSITION'].data.shape[0]


    idx = np.where(hdu['WHT'].data==0)
    hdu['SCI'].data[idx] = 0
        
    #get append tables
    table_list = ['POSITION','SHAPE','FLUX','PHYSICAL','DETECTION','PHOTOMETRY'] 
    for t in table_list:  #loop over tables
      noff = 0
      nrcurr = 0

      #get the table from each tile
      #for i in range(args.start,args.end,1):
      for i,fin in enumerate(flist):
        #open injected image
        fname = f'{args.fdir_embed_subimages}/{fin}'
        hdu_data_injection = fits.open(fname)

        if(i==0): #create table if needed
          hdu_table = fits.BinTableHDU.from_columns(hdu_data_injection[t].columns, nrows=nrows, name=t)

        nrcurr = hdu_data_injection[t].data.shape[0]
        #if('id' in hdu_data_injection[t].data.names):
        #  hdu_data_injection[t].data['id'] += 10000*i 
        for colname in hdu_data_injection[t].columns.names:
            hdu_table.data[colname][noff:noff+nrcurr] = hdu_data_injection[t].data[colname]
        noff+=nrcurr

      #append new long table onto the output hdu
      hdu.append(hdu_table)

    #write out image with injected images included
    print(f'Writing output image {args.output}.')
    hdu.writeto(args.output,overwrite=True)

    #end the time
    time_global_end = time.time()
    print(f'Total time for script = {time_global_end-time_global_start}.')


if __name__=="__main__":
  main()
