#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import numpy as np

from astropy.io import fits

parser = ArgumentParser()
parser.add_argument("--sub_image", type=str, default="small_image.fits",
                    help="Full path to the sub-image FITS file")
parser.add_argument("--full_image", type=str, default="giant_image.fits",
                    help=("Full path to the output image.  If full_header is supplied this"
                          "file must not already exist (we do not overwrite for safety)"))
parser.add_argument("--sub_header", type=str, default=None,
                    help=("Optional, path to a file defining the sub_image header.",
                          "If not given, take directly from the SCI extension of the sub_image."))
parser.add_argument("--full_header", type=str, default="hlf_v2.0.1_30mas_cropped.hdr",
                    help=("Optional, path to a file defining the full_image header."
                          "If not given, take directly from the SCI extension of the full_image,"
                          "which must already exist."))


  #0  PRIMARY       1 PrimaryHDU      37   ()
  #1  SCI           1 ImageHDU        41   (2048, 2048)   float32
  #2  POSITION      1 BinTableHDU     20   81R x 5C   [K, D, D, D, D]
  #3  SHAPE         1 BinTableHDU     22   81R x 5C   [K, D, D, D, D]
  #4  FLUX          1 BinTableHDU     34   81R x 11C   [K, D, D, D, D, D, D, D, D, D, D]
  #5  PHYSICAL      1 BinTableHDU     22   81R x 6C   [K, D, D, D, D, D]
  #6  DETECTION     1 BinTableHDU     17   81R x 4C   [K, L, D, D]
  #7  PHOTOMETRY    1 BinTableHDU     51   81R x 21C   [K, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D]


def find_slices(sub_header, full_header):
    """
    Parameters
    ----------
    sub_header : fits.Header() or dict
        Header of the sub_image, including CD, CRVAL, CRPIX keywords

    full_header : fits.Header() or dict
        Header of the superimage, including CD, CRVAL, CRPIX keywords

    Returns
    -------
    xinds : slice instance
        Slice into AXIS1 of the full image corresponding to the sub_image.

    yinds : slice instance
        Slice into AXIS2 axis of the full image corresponding to the sub_image
    """

    large = full_header
    small = sub_header

    same = ["PC1_1", "PC2_2", "PC1_2", "PC2_1"]
    for k in same:
        if(k not in large):
            print(f"Keyword {k} not present in full header.")
        else:
            print(f"large k {k} {large[k]}")
        if(k not in small):
            print(f"Keyword {k} not present in sub header.")
        else:
            print(f"small k {k} {small[k]}")
    #exit()
    same = ["CD1_1", "CD2_2", "CD1_2", "CD2_1", "CRVAL1", "CRVAL2"]
    for k in same:
        if(k not in large):
            print(f"Keyword {k} not present in full header.")
        if(k not in small):
            print(f"Keyword {k} not present in sub header.")
    for k in same[4:]:
        assert np.allclose(large[k], small[k], rtol=1e-6), f"{k} is not the same in current and target header"

    tsize = small["NAXIS1"], small["NAXIS2"]
    #xstart = large["CRPIX1"] - small["CRPIX1"]
    #ystart = large["CRPIX2"] - small["CRPIX2"]
    xstart = large["CRPIX2"] - small["CRPIX2"]
    ystart = large["CRPIX1"] - small["CRPIX1"]
    assert np.mod(xstart, 1.0) == 0.0
    assert np.mod(ystart, 1.0) == 0.0
    xstart = int(xstart)
    ystart = int(ystart)

    #xend = xstart + tsize[0]
    #yend = ystart + tsize[1]
    xend = xstart + tsize[1]
    yend = ystart + tsize[0]

    # make sure sub image really fits in the super image
    print(f"large NAXIS1 {large['NAXIS1']} NAXIS2 {large['NAXIS2']}")
    print(f"small NAXIS1 {small['NAXIS1']} NAXIS2 {small['NAXIS2']}")
    print(f"large CRPIX1 {large['CRPIX1']} CRPIX2 {large['CRPIX2']}")
    print(f"small CRPIX1 {small['CRPIX1']} CRPIX2 {small['CRPIX2']}")
    print(f"large CRVAL1 {large['CRVAL1']} CRVAL2 {large['CRVAL2']}")
    print(f"small CRVAL1 {small['CRVAL1']} CRVAL2 {small['CRVAL2']}")
    print(f"xstart {xstart} ystart {ystart}")
    assert xstart >= 0
    assert ystart >= 0
    #assert xend < large["NAXIS1"]
    #assert yend < large["NAXIS2"]
    assert yend <= large["NAXIS1"]
    assert xend <= large["NAXIS2"]

    xinds = slice(xstart, xend)
    yinds = slice(ystart, yend)

    print(f"xinds {xinds}")
    print(f"yinds {yinds}")

    flag_transpose=False
    if('PC1_1' in large):
        if(large['PC1_1']*large['PC2_2']<0):
            flag_transpose = True

    flag_transpose=False
    return xinds, yinds, flag_transpose
    #return yinds, xinds, flag_transpose


def empty_image(hdr, subh, extensions, fill, primary_hdr=None):
    size = hdr["NAXIS2"], hdr["NAXIS1"]

    subh_rs = subh.copy()
    subh_rs['NAXIS1'] = hdr['NAXIS1']
    subh_rs['NAXIS2'] = hdr['NAXIS2']
    subh_rs['CRPIX1'] = hdr['CRPIX1']
    subh_rs['CRPIX2'] = hdr['CRPIX2']

    # Make a zeroed full size image

    hdul = fits.HDUList()

    if(primary_hdr is not None):
        pri = fits.PrimaryHDU(header=primary_hdr)
        hdul = fits.HDUList(hdus=[pri])
        #hdul.writeto('test_primary.fits')
    else:
        hdul = fits.HDUList()

    for e, f in zip(extensions, fill):
        plane = np.ones(size) * f
        #hdul.append(fits.ImageHDU(plane, header=hdr))
        hdul.append(fits.ImageHDU(plane, header=subh_rs))
        hdul[-1].header['EXTNAME'] = e

    if(primary_hdr is not None):
        hdul[0].header['EXTNAME'] = 'PRIMARY'
    return hdul


def combine(outvals, invals, xinds, yinds, flag_transpose):
    """This just sums with the existing image. In principle this could do more
    complicated things (like average or weighted average)
    """
    if(flag_transpose==False):
        outvals.data[xinds, yinds] += invals.data[:]
    else:
        outvals.data[xinds, yinds] += invals.data[:].T
    return outvals


if __name__ == "__main__":

    #extensions = ["SCI", "ERR", "EXP", "WHT"]
    #fill = [0, 0, 0, 0]
    extensions = ["SCI"]
    fill = [0]
    args = parser.parse_args()

    print(f"args.full_header {args.full_header}")
    print(f"args.full_image  {args.full_image}")
    print(f"args.sub_header  {args.sub_header}")
    print(f"args.sub_image   {args.sub_image}")

    if args.full_header:
        assert not os.path.exists(args.full_image)
        # set up the output image
        #fullh = fits.Header.fromfile(args.full_header,'SCI')
        primary_header = fits.getheader(args.full_header,'PRIMARY')
        if(primary_header['TELESCOP']=='JWST'):
            fullh = fits.getheader(args.full_header,'SCI')
        else:
            fullh = primary_header.copy()
    elif os.path.exists(args.full_image):
        print(f'Existing Full image = {args.full_image}')
        exit()
        full_image = fits.open(args.full_image, mode="update")
        fullh = full_image["SCI"].header
    else:
        print(f"Full image {args.full_image} does not exist, and no header for new file specified in --full_header")

    if args.sub_header:
        subh = fits.Header.fromfile(args.sub_header)
    else:
        subh = fits.getheader(args.sub_image, "SCI")

    print(repr(fullh))

    # note the order of y, x here.  yinds refers to NAXIS1 which is the second
    # python axis
    yinds, xinds, flag_transpose = find_slices(subh, fullh)

    # now open the sub_image and 'combine' with or add to the full image
    with fits.open(args.sub_image) as sub_image:


        # this *could* just open an existing full size image and operate on it
        primary_header = fits.getheader(args.sub_image,'PRIMARY')
        full_image = empty_image(fullh, subh, extensions, fill, primary_hdr=primary_header)
        print(f"Data type check {type(full_image['SCI'].data[0,0])} {(type(full_image['SCI'].data[0,0])==np.float64)}")

        for e in extensions:
            if(type(full_image[e].data[0,0])==np.float64):
                full_image[e].data = full_image[e].data.astype(np.float32)
            if(type(sub_image[e].data[0,0])==np.float64):
                sub_image[e].data = sub_image[e].data.astype(np.float32)
            combine(full_image[e], sub_image[e], yinds, xinds, flag_transpose)

    # at this point one should probably probably propogate lots of header
    # keywords, taking care not to overwrite existing ones
    for hdu in full_image:
        hdu.header.extend(subh, unique=True)

    #write the image
    full_image.writeto(args.full_image)

   
    #full_image = fits.open(args.full_image,mode='append')
    #here we add the catalog extensions
    table_list = ['POSITION','SHAPE','FLUX','PHYSICAL','DETECTION','PHOTOMETRY']
    
    with fits.open(args.sub_image) as sub_image:
      for i,t in enumerate(table_list):
        print(f'table {t}')
        fits.append(args.full_image,sub_image[t].data,header=sub_image[t].header)
