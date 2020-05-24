@echo off
setlocal enableextensions enabledelayedexpansion

REM ---------------------------------------------------------------------
REM
REM  Script for aligning averaged HDR images (NRGF- and Sobel-filtered)
REM  of multiple sites (inter-site alignment).
REM
REM  The resulting aligned FITS and BMP images and diagnostic data
REM  are written to the indicated output directory.
REM
REM ---------------------------------------------------------------------

REM --------------- USER-CONFIGURABLE PARAMETERS ------------------------

REM -----------
REM
REM Program's executable file.
REM
REM -----------
set CATETOOL=catetool.exe

REM -----------
REM
REM List of input files in alignment order.
REM
REM -----------
set INPUT_FILES=^
    "T:\CATE\NRGFs\001_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\002_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\005_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\006_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\007_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\000_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\009_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\010_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\011_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\012_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\014_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\018_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\023_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\024_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\028_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\029_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\035_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\039_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\041_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\045_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\048_averaged_sobel.fits" ^
    "T:\CATE\NRGFs\051_averaged_sobel.fits"


REM -----------
REM
REM Output directory.
REM
REM -----------
set OUTPUT_DIR=output

REM ---------------------------------------------------------------------


if not exist %OUTPUT_DIR% (
    mkdir %OUTPUT_DIR%
)

%CATETOOL% --mode multi-site --log_level verbose --output_dir %OUTPUT_DIR% --input_files %INPUT_FILES%
