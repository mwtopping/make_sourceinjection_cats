#!/bin/bash
#SBATCH --job-name=si_complete # Job name
#SBATCH --partition=comp-astro # Partition name
#SBATCH --account=comp-astro # Account name
#SBATCH --ntasks=40             # Number of MPI ranks
#SBATCH --nodes=1               # Total number of nodes requested
#SBATCH --ntasks-per-node=40    # How many tasks on each node
#SBATCH --time=24:00:00         # Time limit (hh:mm:ss)
#SBATCH --output=si_completeness_%j.log     # Standard output and error log

module purge
module load cuda11.2
module load python/3.8.6
module load slurm
module list


# Select a region to model
# Generate headers for tiles
sh run_initialize_generate_headers.sh
cd generate_headers
sh run_generate_headers.sh
cd ..

# Generate multiband images from mock catalog
#python3 initialize_source_injection.py
sh run_initialize_source_injection.sh
cd generate_images
sh run_generate_images.sh
cd ..

# Embed generated images
sh run_initialize_embed_subimages.sh
cd embed_subimages
sh run_embed_subimages.sh
cd ..

# Add generated images to filter mosaics
sh run_initialize_insert_images.sh
cd insert_images
sh run_insert_images.sh
cd ..

# Extract layer of f200w
sh run_initialize_extract_layer.sh
cd extract_layers
sh run_extract_layers.sh
cd ..

# Create detection image from the stack
sh run_initialize_make_detection_image.sh
cd make_detection_image
sh run_make_detection_image.sh
cd ..

# Perform detection
sh run_initialize_detection.sh
cd detection
sh run_detection.sh
cd ..

# record completeness in output catalog
sh run_initialize_completeness.sh
cd completeness
sh run_completeness.sh
cp completeness.fits ..
cd ..
python3 check_completeness.py
sh clean.sh
