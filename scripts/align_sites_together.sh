#
#  Script for aligning averaged HDR images (NRGF- and Sobel-filtered)
#  of multiple sites (inter-site alignment).
#
#  The resulting aligned FITS and BMP images and diagnostic data
#  are written to the indicated output directory.
#

# --------------- USER-CONFIGURABLE PARAMETERS ------------------------

#
# Program's executable file.
#
CATETOOL=target/release/catetool

#
# List of input files in alignment order.
#
INPUT_FILES=("/mnt/depot/xfer/CATE/NRGFs/001_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/002_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/005_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/006_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/007_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/000_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/009_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/010_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/011_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/012_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/014_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/018_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/023_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/024_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/028_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/029_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/035_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/039_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/041_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/045_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/048_averaged_sobel.fits" \
             "/mnt/depot/xfer/CATE/NRGFs/051_averaged_sobel.fits")

#
# Output directory.
#
OUTPUT_DIR=output

# ---------------------------------------------------------------------

mkdir -p $OUTPUT_DIR

$CATETOOL --mode multi-site --output_dir $OUTPUT_DIR --log_level verbose --input_files "${INPUT_FILES[@]}"
