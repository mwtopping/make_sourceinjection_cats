import numpy as np
import argparse
from astropy.io import fits
from astropy.wcs import WCS

#########################################
# Routine to parse command line arguments
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Detection flags and options from user.")

    parser.add_argument("--input", type=str,
                default="input.fits",
                help='Input fits file to add sci layer to. (default: input.fits)')

    parser.add_argument("--sources", type=str,
                default="sources.fits",
                help='Fits with sources add. (default: sources.fits)')

    parser.add_argument("--output", type=str,
                default="output.fits",
                help='Output fits file. (default: sources.fits)')

    parser.add_argument("--x", type=int,
                default=0,
                help='x offset for insertion. (default: 0)')

    parser.add_argument("--y", type=int,
                default=0,
                help='y offset for insertion. (default: 0)')

    parser.add_argument('-v', '--verbose',
                dest='verbose',
                action='store_true',
                help='Print helpful information to the screen? (default: False)',
                default=False)

    return parser


######################################
# main function
######################################
def main():

    #make the parser
    parser = create_parser()

    # read args
    args = parser.parse_args()

    if(args.verbose):
        print(f"x offset = {args.x}")
        print(f"y offset = {args.y}")

    #load hdu
    hdui = fits.open(args.input)

    #load sources
    sdata = fits.getdata(args.sources,'SCI')
    shdu  = fits.open(args.sources)

    #note that we assume sdata is in units of nJy here
    #so we need to convert to MJy/sr
    wcs = WCS(hdui[1].header)
    pixel_area = np.abs(np.linalg.det(wcs.pixel_scale_matrix*3600))
    flux_to_nJy  = 1e15 / 4.25e10 * pixel_area #header_flux['PIXAR_SR']*1.e15 # to nJy
    sdata /= flux_to_nJy  #inverse from nJy to MJy/sr

    #add sources
    xmax = np.min([args.x + sdata.shape[1],hdui[1].data.shape[1]])
    ymax = np.min([args.y + sdata.shape[0],hdui[1].data.shape[0]])
    hdui[1].data[args.y:ymax,args.x:xmax] += sdata

    print(f"Writing to {args.output}.")
    hdui.writeto(args.output,overwrite=True)

    print(f'Re-writing ra, dec of sources after insertion.')
    sdata*=flux_to_nJy
    x = shdu[3].data['x'] + args.x
    y = shdu[3].data['y'] + args.y
    ra, dec = wcs.wcs_pix2world(x,y,0)
    shdu[3].data['ra']  = ra
    shdu[3].data['dec'] = dec
    shdu.writeto(args.sources,overwrite=True)
  
    print("Finished!") 
    


#################################3
# run
#################################3
if __name__=="__main__":
    main()
