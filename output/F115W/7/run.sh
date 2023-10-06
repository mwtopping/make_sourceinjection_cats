# Select a region to model
# Generate headers for tiles
#sh run_initialize_generate_headers.sh
#cd generate_headers
#sh run_generate_headers.sh
#cd ..

# Generate multiband images from mock catalog
sh run_initialize_source_injection.sh
cd generate_images
sh run_generate_images.sh
cd ..

# Embed generated images
#sh run_initialize_embed_subimages.sh
#cd embed_subimages
#sh run_embed_subimages.sh
#cd ..

# Add generated images to filter mosaics
#sh run_initialize_insert_images.sh
#cd insert_images
#sh run_insert_images.sh
#cd ..

# Extract layer of f200w
#sh run_initialize_extract_layer.sh
#cd extract_layers
#sh run_extract_layers.sh
#cd ..

# Create detection image from the stack
#sh run_initialize_make_detection_image.sh
#cd make_detection_image
#sh run_make_detection_image.sh
#cd ..

# Perform detection
#sh run_initialize_detection.sh
#cd detection
#sh run_detection.sh
#cd ..

# record completeness in output catalog
#sh run_initialize_completeness.sh
#cd completeness
#sh run_completeness.sh
#cp completeness.fits ..
#cd ..
#python3 check_completeness.py
#sh clean.sh
