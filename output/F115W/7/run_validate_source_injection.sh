#!/bin/bash
#python3 jades_validate_source_injection.py --path /Users/brant/github/jades-pipeline --input output/training_data_0.fits -c 0.2 0.3 0.5 0.6 0.7 1.0 -v
#python3 jades_validate_source_injection.py --path /Users/brant/github/jades-pipeline --input output/training_data_0.fits -c 0.2 0.3 0.5 0.6 0.7 1.0 -v -t 0.01
python3 jades_validate_source_injection.py --path /Users/brant/github/jades-pipeline --input output/training_data_0.fits -c 0.2 0.3 0.5 0.6 0.7 1.0 -v -t 0.03 --output validation_catalog.txt
python3 make_colorized_segmap.py segmap_validation.fits
