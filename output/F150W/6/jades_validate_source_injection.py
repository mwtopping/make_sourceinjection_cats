import sys
import argparse
import time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from photutils.segmentation import SourceCatalog
from scipy.spatial import KDTree


#########################################
# Routine to parse command line arguments
#########################################

def create_parser():

    # Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Detection flags and options from user.")


    parser.add_argument("--input", type=str,
                        default="input.fits")


    parser.add_argument("--output", type=str,
                        default="output.txt")


    parser.add_argument('-p','--path',
                default='./',
                type=str,
                help='Path to jades-pipeline directory.') 

    parser.add_argument('--output_segmap',
                default='segmap_validation.fits',
                type=str,
                help='Output segmap from validation test.') 

    parser.add_argument('-t','--threshold',
                default=3.0,
                type=float,
                help='Threshold value for object detection.')

    parser.add_argument('-np','--npix',
                default=5.0,
                type=float,
                help='Number of pixels above threshold for object detection.')

    parser.add_argument('--centroid',
        default='centroid_win',
        type=str,
        help='Centroiding method, default windowed.')

    parser.add_argument('-c',
        '--circ_aper',
        nargs='+',
        type=float,
        help='list of aperture diameters (in arcsec)')

    parser.add_argument('--nmad',
                dest='nmad',
                action='store_true',
                help='Compute nmad errors? (default: False -- dummy for validation)',
                default=False)


    parser.add_argument('--bkgsub',
                dest='bkgsub',
                action='store_true',
                help='Perform background subtraction? (default: False -- dummy for validation)',
                default=False)

    parser.add_argument('-v', '--verbose',
                dest='verbose',
                action='store_true',
                help='Print helpful information to the screen? (default: False)',
                default=False)

    return parser


#########################################
# Routine to define columns in cat output
#########################################
def DefineColumns(args,flag_val=False):

    #define apertures
    apertures = args.circ_aper

    #define columns
    if(args.nmad==False):
        if(args.bkgsub):
            columns = ['label','xcentroid','ycentroid','area','semimajor_sigma','semiminor_sigma','orientation','eccentricity','min_value','max_value','local_background','segment_flux','segment_fluxerr','kron_flux','kron_fluxerr','kron_s_flux','kron_s_fluxerr','kron_radius','fwhm','gini','RHALF','R0_flux','R0_fluxerr','R0_bkgsub','bkg']
        else:
            columns = ['label','xcentroid','ycentroid','area','semimajor_sigma','semiminor_sigma','orientation','eccentricity','min_value','max_value','local_background','segment_flux','segment_fluxerr','kron_flux','kron_fluxerr','kron_s_flux','kron_s_fluxerr','kron_radius','fwhm','gini','RHALF','R0_flux','R0_fluxerr']

    else:
        #if computing nmad errors, add corresponding
        #columns to the table
        if(args.bkgsub):
            columns = ['label','xcentroid','ycentroid','area','semimajor_sigma','semiminor_sigma','orientation','eccentricity','min_value','max_value','local_background','segment_flux','segment_fluxerr','kron_flux','kron_fluxerr','kron_fluxerr_nmad','kron_s_flux','kron_s_fluxerr','kron_s_fluxerr_nmad','kron_radius','fwhm','gini','RHALF','R0_flux','R0_fluxerr','R0_fluxerr_nmad','R0_bkgsub','bkg']
        else:
            columns = ['label','xcentroid','ycentroid','area','semimajor_sigma','semiminor_sigma','orientation','eccentricity','min_value','max_value','local_background','segment_flux','segment_fluxerr','kron_flux','kron_fluxerr','kron_fluxerr_nmad','kron_s_flux','kron_s_fluxerr','kron_s_fluxerr_nmad','kron_radius','fwhm','gini','RHALF','R0_flux','R0_fluxerr','R0_fluxerr_nmad']


    for i in range(len(apertures)):
        r_flux = f'R{i+1}_flux'
        columns.append(r_flux)
        r_fluxerr = f'R{i+1}_fluxerr'
        columns.append(r_fluxerr)
        
        #if computing nmad errors, add a column
        #for each flux measurement
        if(args.nmad==True):
            r_fluxerr_nmad = f'R{i+1}_fluxerr_nmad'
            columns.append(r_fluxerr_nmad)

        if(args.bkgsub):
            r_bkgsub = f'R{i+1}_bkgsub'
            columns.append(r_bkgsub)

    #need to add exposure tims
    #if(args.nmad==True):
    #texp = 't_exp'
    #columns.append(texp)
    #wht = 'wht'
    #columns.append(wht)

    if(args.centroid!='centroid'):
        columns.append('xbarycenter')
        columns.append('ybarycenter')

    if(flag_val):
        columns.append('distance')
        columns.append('index')
        columns.append('x_in')
        columns.append('y_in')
        columns.append('flux_in')
        columns.append('pa_in')
        columns.append('q_in')
        columns.append('rhalf_in')
        columns.append('sersic_in')
        columns.append('noise_in')
        columns.append('snr_in')

    #return the columns
    return columns



#########################################
# Main function
#########################################
def main():

    #make the parser
    parser = create_parser()

    # read args
    args = parser.parse_args()

    #what's the input
    if(args.verbose):
        print(f"Input noiseless source injection image: {args.input}.")
        print(f"Detection threshold {args.threshold}")
        print(f"Centroiding method? {args.centroid} [Default is sextractor windowed centroids]")

    #load the image
    hdu = fits.open(args.input)
    flux = hdu['SCI'].data.copy()

    # see if code path is set
    sys.path.append(f'{args.path}')
    sys.path.append(f'{args.path}/detection/')
    sys.path.append(f'{args.path}/detection/detection/')
    import jades_photutils_interface as jpui
    import quicklook as jql
    import detect_and_mask as dam

    #perform detection
    flux_err = np.full_like(flux, 0.5) # just assume something
    snr = flux/flux_err
    cat, data, segmap, data_conv = jpui.CreateBlendedSourceCatalog(snr,threshold=args.threshold,npixels=args.npix,verbose=args.verbose)

    #save the segmap
    fits.writeto(args.output_segmap,data=segmap,header=hdu['SCI'].header,overwrite=True)

    #compute centroids
    if(args.centroid!='centroid'):

        print(f"Computing centroids....")
        #Perform centroids
        time_start = time.time()

        xbarycenter = cat.xcentroid.copy()
        ybarycenter = cat.ycentroid.copy()
        xcentroid_win = cat.xcentroid_win
        ycentroid_win = cat.ycentroid_win

        cat.xcentroid = xcentroid_win
        cat.ycentroid = ycentroid_win
        cat.centroid[:,0] = xcentroid_win
        cat.centroid[:,1] = ycentroid_win

        #for i in range(len(xcentroid_win)):
        for i in range(100):
            if(xbarycenter[i]!=xcentroid_win[i]):
                print(f'i {i} id {cat.label[i]} xcentroid {cat.xcentroid[i]} ycentroid {cat.ycentroid[i]} xwin {xcentroid_win[i]} ywin {ycentroid_win[i]} barycenters x {xbarycenter[i]} y {ybarycenter[i]}')


        ## how long did this take?
        time_end = time.time()
        print(f"Time to compute centroid from catalog images = {time_end-time_start}")


    #create photometric catalog
    mask = np.zeros_like(flux, dtype=bool)
    cat_phot = SourceCatalog(flux, segmap, error=flux_err, mask=mask, detection_cat=cat)

    if(args.verbose):
        print(f"Performing photometry step...")

    #measure half light radius
    cat_phot.fluxfrac_radius(0.5,name='RHALF')


    ##################################################
    #perform the R0 photometry
    ##################################################
    r_ee = 0.3
    pixel_scale = 0.03
    circ_flux, circ_flux_err = cat_phot.circular_photometry(r_ee/pixel_scale,name='R0')
    print(f"circ_flux shape {circ_flux.shape}")


    #perform aperture photometry
    r_aper = []
    #record the aperture
    r_aper.append(r_ee) #80% ee aperture in radius

    ##################################################
    #perform a loop of circular aperture photometry
    ##################################################

    for i,d in enumerate(args.circ_aper):

        r = d * 0.5 #convert diameter to radii in arc sec

        #name the radius
        rname = f'R{i+1}'

        #perform aperture photometry
        circ_flux, circ_flux_err = cat_phot.circular_photometry(r/pixel_scale,name=rname)

        if(args.bkgsub):

            print(f"Performing background estimation for {r}...")
            area = np.pi*(r/pixel_scale)**2
            flux_bkg = circ_flux - area*bkg_per_pixel
            cat_phot.add_extra_property(rname+'_bkgsub',flux_bkg)

        #if computing nmad errors, add an entry in the catalog
        if(args.nmad):
            cat_phot.add_extra_property(rname+'_fluxerr_nmad',np.zeros(cat_phot.nlabels))



        #record aperture
        r_aper.append(r)

        #record aperture correction
        '''
        if(args.acoff==False):
            if(flag_inst=='HST'):
              aci = hsta.aperture_correction(PSF_FILTER,r) #input radius in arcsec
            elif(flag_inst=='NIRCAM'):
              aci = nca.aperture_correction(PSF_FILTER,r) #input radius in arcsec
            elif(flag_inst=='MIRI'):
              aci = ma.aperture_correction(PSF_FILTER,r) #input radius in arcsec
            ac.append(aci)

            if(args.verbose):
              print(f'R = {r} aperture correction for psf filter {PSF_FILTER} is ac = {aci}')
        '''




    #kron s
    cat_phot.kron_photometry((1.2, 1.4, 0.0), name='kron_s')


    #add barycenters
    if(args.centroid!='centroid'):

        cat_phot.add_extra_property('xbarycenter',xbarycenter)
        cat_phot.add_extra_property('ybarycenter',ybarycenter)

    #if nmad errors are used, we need to add them
    #to the catalog
    if(args.nmad):
        cat_phot.add_extra_property('R0_fluxerr_nmad',np.zeros(cat_phot.nlabels))
        cat_phot.add_extra_property('kron_fluxerr_nmad',np.zeros(cat_phot.nlabels))
        cat_phot.add_extra_property('kron_s_fluxerr_nmad',np.zeros(cat_phot.nlabels))


    ##################################################
    #define columns for photometry catalog
    ##################################################
    columns = DefineColumns(args)

    ##################################################
    #create a table from the photometry catalog
    ##################################################
    tbl = cat_phot.to_table(columns=columns) 


    ###
    # find closest match for each object from the input catalog
    ###

    #make a tree from the input
    cat_val = hdu['CAT'].data
    print(cat_val.names)

    #how many
    print(f"Number of found / input sources = {len(cat_phot)}/{len(cat_val)}.")


    tree = KDTree(np.asarray([cat_val['x'],cat_val['y']]).T)
    dis, idx = tree.query(np.asarray([tbl['xcentroid'],tbl['ycentroid']]).T,k=1)

    cat_phot.add_extra_property('distance',dis)
    cat_phot.add_extra_property('index',idx)
    cat_phot.add_extra_property('x_in',cat_val['x'][idx])
    cat_phot.add_extra_property('y_in',cat_val['y'][idx])
    cat_phot.add_extra_property('flux_in',cat_val['fclear'][idx])
    cat_phot.add_extra_property('snr_in',cat_val['snr'][idx])
    cat_phot.add_extra_property('noise_in',cat_val['noise'][idx])
    cat_phot.add_extra_property('pa_in',cat_val['pa'][idx])
    cat_phot.add_extra_property('q_in',cat_val['q'][idx])
    cat_phot.add_extra_property('rhalf_in',cat_val['rhalf'][idx])
    cat_phot.add_extra_property('sersic_in',cat_val['sersic'][idx])
    cat_phot.add_extra_property('npix_in',cat_val['n_pix'][idx])

    columns = DefineColumns(args,flag_val=True)
    tbl = cat_phot.to_table(columns=columns) 

    ##################################################
    #save table to ascii file
    ##################################################

    if(args.verbose):
        print(f"Writing photometry catalog to {args.output}.")
    tbl.write(args.output,overwrite=True,format='ascii')


#########################################
# Run the program
#########################################
if __name__=="__main__":
    main()