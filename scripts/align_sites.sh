#
#  Script for aligning single site HDR images (intra-site alignment),
#  multiple sites at a time.
#
#  The resulting averaged FITS, reference blocks and diagnostic data
#  are written to the indicated output directory.
#

# --------------- USER-CONFIGURABLE PARAMETERS ------------------------

#
# Program's executable file.
#
CATETOOL=target/release/catetool

#
# Directory with input files (single-site partial HDRs). The script expects
# there are subdirectories 000, 001 etc. for each site.
#
INPUT_DIR=/mnt/depot/xfer/CATE/unfiltered_HDRs

#
# Output directory.
#
OUTPUT_DIR=output

#
# Text file containing the following site information in each line:
#
#   site_number ref_pos_x ref_pos_y moon_diameter
#
# where "ref_pos" is the position of the base of the large prominence in the first image.
#
SITE_LIST=scripts/all_sites.txt

# ---------------------------------------------------------------------
set -e

cat $SITE_LIST | while read line; do
    chomped_line=${line%$'\r'} # remove the carriage return in case the file has been edited under Windows
    tokens=($chomped_line)

    site=${tokens[0]}
    x=${tokens[1]}
    y=${tokens[2]}
    moon_diameter=${tokens[3]}

    if [ "$site" = "056" ]; then
        set_background_threshold="--background_threshold 1100.0"
    elif [ "$site" = "022" ]; then
        set_background_threshold="--background_threshold 1800.0"
    elif [ "$site" = "032" ]; then
        set_background_threshold="--background_threshold 1800.0"
    fi

    mkdir -p $OUTPUT_DIR/$site

    $CATETOOL \
        --mode single-site \
        --save_aligned no \
        --exclude_moon_diameter $moon_diameter \
        $set_background_threshold \
        --output_avg_file $OUTPUT_DIR/"$site"_averaged.fits \
        --output_dir $OUTPUT_DIR/$site \
        --input_files $(ls $INPUT_DIR/$site/*.fits) \
        --ref_pos $x $y \
        --log_level verbose

    mv $OUTPUT_DIR/$site/diagnostic_data.txt $OUTPUT_DIR/"$site"_diagnostic_data.txt
done
