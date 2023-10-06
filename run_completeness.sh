python3 jades_completeness.py --input /data/groups/comp-astro/brant/source_injection/automation/make_detection_image/snr.injected.fits  --cat /data/groups/comp-astro/brant/source_injection/automation/detection/det.cat --segmap /data/groups/comp-astro/brant/source_injection/automation/detection/segmap.fits --output completeness.fits
python3 check_completeness.py
