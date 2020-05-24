@echo off
setlocal enableextensions enabledelayedexpansion

REM ---------------------------------------------------------------------
REM
REM  Script for aligning single site HDR images (intra-site alignment),
REM  multiple sites at a time.
REM
REM  The resulting averaged FITS, reference blocks and diagnostic data
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
REM Directory with input files (single-site partial HDRs). The script expects
REM there are subdirectories 000, 001 etc. for each site.
REM
REM -----------
set INPUT_DIR=T:\CATE\unfiltered_HDRS

REM -----------
REM
REM Output directory.
REM
REM -----------
set OUTPUT_DIR=output

REM -----------
REM
REM Text file containing the following site information in each line:
REM
REM   site_number ref_pos_x ref_pos_y moon_diameter
REM
REM
REM where "ref_pos" is the position of the base of the large prominence in the first image.
REM
REM -----------
set SITE_LIST=all_sites.txt

REM ---------------------------------------------------------------------


for /f "tokens=1,2,3,4" %%i in (%SITE_LIST%) do (
    set SITE=%%i
    set REF_X=%%j
    set REF_Y=%%k
    set MOON_DIAMETER=%%l

    echo ---------- Aligning site !SITE!

    if not exist %OUTPUT_DIR%\!SITE! (
        mkdir %OUTPUT_DIR%\!SITE!
    )

    call :align_one_site !SITE! !REF_X! !REF_Y! !MOON_DIAMETER!
)

goto :eof

REM -----------------------------------

:align_one_site

set SITE=%1
set REF_X=%2
set REF_Y=%3
set MOON_DIAMETER=%4

if %SITE% == 056 (
    set SET_BACKGROUND_THRESHOLD="--background_threshold 1100.0"
) else (
if %SITE% == 022 (
    set SET_BACKGROUND_THRESHOLD="--background_threshold 1800.0"
) else (
if %SITE% == 032 (
    set SET_BACKGROUND_THRESHOLD="--background_threshold 1800.0"
)
)
)

set CMD_LIST_INPUT_FILES=dir %INPUT_DIR%\%SITE% /O:N /B
set INPUT_FILES=
for /f "usebackq tokens=*" %%a in (`%CMD_LIST_INPUT_FILES%`) do (
    set INPUT_FILES=!INPUT_FILES! %INPUT_DIR%\%SITE%\%%a
)

%CATETOOL% ^
    --mode single-site ^
    --output_dir %OUTPUT_DIR%\%SITE% ^
    --input_files %INPUT_FILES% ^
    --output_avg_file %OUTPUT_DIR%/%SITE%_averaged.fits ^
    --ref_pos %REF_X% %REF_Y% ^
    --log_level verbose ^
    --exclude_moon_diameter %MOON_DIAMETER% ^
    %SET_BACKGROUND_THRESHOLD%

move %OUTPUT_DIR%\%SITE%\diagnostic_data.txt %OUTPUT_DIR%\%SITE%_diagnostic_data.txt

goto :eof
