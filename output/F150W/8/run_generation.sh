#!/bin/bash
#python3 jades_generate_images.py --config_file config.yml --output_dir output/ --psf psf/F200W_ePSF.fits --seed 1337
#python3 jades_generate_images.py --config_file config.yml --output_dir output/ --psf psf/F200W_ePSF.fits --seed 1337 --grid --ngrid_x 5 --ngrid_y 5
python3 jades_generate_images.py --config_file config.yml --output_dir output/ --psf psf/F200W_ePSF.fits --seed 1337 --grid --ngrid_x 10 --ngrid_y 10
