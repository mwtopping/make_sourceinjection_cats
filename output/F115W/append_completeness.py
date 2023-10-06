from astropy.io import fits
import numpy as np
import sys
fname = 'completeness_list.txt'
if(len(sys.argv)>1):
    fname = sys.argv[1]

output = 'out.fits'
if(len(sys.argv)>2):
    output = sys.argv[2]

fp = open(fname)
fl = fp.readlines()
fp.close()
fl = [f.strip('\n') for f in fl]




#get append tables
nrows = 0

#get append tables
for ifile, cfile in enumerate(fl):

    hduc = fits.open(cfile)
    nrows += len(hduc['DETECTION'].data)

    if(ifile==0):
        hdu = fits.HDUList(hdus=[hduc['SCI']])
print(f'nrows = {nrows}')

table_list = ['POSITION','SHAPE','FLUX','PHYSICAL','DETECTION','PHOTOMETRY']
for t in table_list:  #loop over tables
    noff = 0
    nrcurr = 0
    for i,fin in enumerate(fl):
      #open injected image
      hdu_data_injection = fits.open(fin)

      if(i==0): #create table if needed
        hdu_table = fits.BinTableHDU.from_columns(hdu_data_injection[t].columns, nrows=nrows, name=t)

      nrcurr = hdu_data_injection[t].data.shape[0]
      for colname in hdu_data_injection[t].columns.names:
        hdu_table.data[colname][noff:noff+nrcurr] = hdu_data_injection[t].data[colname]
      noff+=nrcurr

    #append new long table onto the output hdu
    hdu.append(hdu_table)

#write out image with injected images included
print(f'Writing output image {output}.')
hdu.writeto(output,overwrite=True)
