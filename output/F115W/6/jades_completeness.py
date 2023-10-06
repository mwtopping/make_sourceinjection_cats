import os
import sys
import numpy as np
import argparse
import sep
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from tqdm import tqdm
import scipy.ndimage as ndi
#########################################
## Create a parser for the command line
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Compute completeness from detection segmap.")

    parser.add_argument("--input",
        type=str,
        default='input.fits',
        help='Input image to extract sci layer from.')

    parser.add_argument("--cat",
        type=str,
        default='det.cat',
        help='Detection catalog.')


    parser.add_argument("--segmap",
        type=str,
        default='segmap.fits',
        help='Detection segmentation map.')

    parser.add_argument("--output",
        type=str,
        default='completeness.fits',
        help='Catalog with completeness info.')

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

  #print info
  print(f'Input snr image {args.input}.')
  print(f'Detection segmap {args.segmap}.')
  print(f'Detection catalog {args.cat}.')
  print(f'Output completeness catalog {args.output}.')

  #read SNR image with catalog
  hdu_snr = fits.open(args.input)
  header_snr = hdu_snr['SCI'].header

  header_snr = hdu_snr['SCI'].header
  data_snr   = hdu_snr['SCI'].data
  data_pos   = hdu_snr['POSITION'].data
  data_shape = hdu_snr['SHAPE'].data
  data_det   = hdu_snr['DETECTION'].data

  #open segmap
  hdu_seg = fits.open(args.segmap)
  segmap = hdu_seg['SCI'].data

  #open detection catalog
  det_cat = Table.read(args.cat,format='ascii')

  #wcs of detection image
  wcs = WCS(header_snr)
  #get pixel area and scale
  pixel_area = np.abs(np.linalg.det(wcs.pixel_scale_matrix*3600))
  pixel_scale = pixel_area**0.5

  #sky positions
  ra  = data_pos['ra']
  dec = data_pos['dec']

  #image positions
  x,y = wcs.wcs_world2pix(ra,dec,0)

  #create elliptical mask
  r = 1
  ellipse_mask = np.zeros_like(hdu_snr['SCI'].data,dtype=bool) 

#  ['id', 'detected', 'snr', 'npix']
# ['id', 'ra', 'dec', 'x_tile', 'y_tile']

  #radius, axis ratio, and pa
  rad = data_shape['rhalf']/pixel_scale
  rad[rad<1] = 1.
  rad*=10
  q   = data_shape['q']
  pa  = data_shape['pa']

  print(f'x   min max {np.min(x)} {np.max(x)}')
  print(f'y   min max {np.min(y)} {np.max(y)}')
  print(f'rad min max {np.min(rad)} {np.max(rad)}')
  print(f'q   min max {np.min(q)} {np.max(q)}')
  print(f'pa  min max {np.min(pa)} {np.max(pa)}')

  #get mask
  sep.mask_ellipse(ellipse_mask,x,y,rad,rad*q,pa)

  segmap_ellipse, nobjects = ndi.label(ellipse_mask)

  fits.writeto('test_mask.fits',data=segmap_ellipse,header=header_snr,overwrite=True)

  #apply mask to segmap
  rev_seg = np.zeros_like(segmap)
  rev_seg[ellipse_mask] = segmap[ellipse_mask]

  #write masked segmap
  fits.writeto('masked_segmap.fits',data=rev_seg,header=header_snr,overwrite=True)

  #record the detection catalog
  for i in tqdm(range(len(data_det))):
    
    #if there is a detection
    iy = int(y[i])
    ix = int(x[i])
    iddet = rev_seg[iy,ix]
    #iddet = rev_seg[ix,iy]
    if(iddet>0):
      data_det[i]['id'] = iddet
      data_det[i]['detected'] = 1
      data_det[i]['snr'] = data_snr[iy,ix]
      #data_det[i]['snr'] = data_snr[ix,iy]
      di = np.where(det_cat['label']==iddet)
      data_det[i]['npix'] = det_cat[di]['area']

    else:
      data_det[i]['detected'] = 0
      data_det[i]['snr'] = 0
      data_det[i]['npix'] = 0

  for name in hdu_snr['DETECTION'].data.names:
    hdu_snr['DETECTION'].data[name] = data_det[name]

  xi = np.where(data_det['detected']==0)[0]
  print(f'Objects not detected {len(xi)}.')
  xi = np.where(data_det['detected']==1)[0]
  print(f'Objects detected     {len(xi)}.')
  xi = np.where(hdu_snr['DETECTION'].data['detected']==0)[0]
  print(f'Objects not detected {len(xi)}.')
  xi = np.where(hdu_snr['DETECTION'].data['detected']==1)[0]
  print(f'Objects detected     {len(xi)}.')
  hdu_snr['DETECTION'].data['detected'][xi] = True
  #  hdul.append(position_cat,header=position_hdu.header,name=position_hdu.name)
  #  hdul.append(shape_cat)
  #  hdul.append(flux_cat)
  #  hdul.append(physical_cat)
  #  hdul.append(detection_cat)
  primary = fits.PrimaryHDU(header=header_snr)
  with fits.HDUList([primary]) as hdul:
    hdu_names = ['POSITION','SHAPE','FLUX','PHYSICAL','DETECTION','PHOTOMETRY']
    for name in hdu_names:
      print(f'name = {name}')
      if(name!='DETECTION'):
        data = hdu_snr[name].data.copy()
        head = hdu_snr[name].header
      else:
        data = data_det.copy()
        head = hdu_snr[name].header
        #xi = np.where(hdu_snr['DETECTION'].data['detected']==1)[0]
        xi = np.where(data['detected']==1)[0]
        print(f'Objects detected     {len(xi)}.')
      hdul.append(fits.BinTableHDU(data=data,header=head,name=name))
    

  hdul.writeto(args.output,overwrite=True)
  hdul.info()

# run the script
if __name__=="__main__":
  main()
