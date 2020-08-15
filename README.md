# catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>

version 0.1.3 (2020-08-15)

*This project is licensed under the terms of the MIT license (see the LICENSE file for details).*

----------------------------------------

- 1\. Overview
  - 1\.1\. Limitations
- 2\. Command-line options
- 3\. Scripts
- 4\. Building from source code
  - 4\.1\. Linux, OS X (and other Unices)
  - 4\.2\. MS Windows
- 5\. Change log

----------------------------------------


## 1. Overview

The purpose of `catetool` is to align and average the images obtained by Continental-America Telescopic Eclipse Experiment (CATE), both intra-site (i.e., alignment of HDR substacks from a single CATE site) and inter-site (i.e., alignment of site-averaged stacks with NRFG and Sobel filter applied for the final time-lapse animation).

Intra-site alignment uses block matching (comparing sums of abs. differences of pixel values) to detect translation of the reference image fragment (a prominence). Inter-site alignment likewise checks pixel differences while iteratively refining relative translation and rotation between site images.


### 1.1 Limitations

For certain sites, intra-site alignment works only for a subset of images:

  - 011: `hdr_011_066_073.fits` - `hdr_011_482_489.fits`
  - 022: `hdr_022_025_032.fits` - `hdr_022_313_320.fits`
  - 025: `hdr_025_126_133.fits` - `hdr_025_542_549.fits`
  - 027: `hdr_027_481_488.fits` - `hdr_027_937_944.fits`
  - 032: `hdr_032_147_154.fits` - `hdr_032_315_322.fits`
  - 043: `hdr_043_130_137.fits` - `hdr_043_530_537.fits`
  - 044: `hdr_044_001_008.fits` - `hdr_044_497_504.fits`
  - 047: `hdr_047_001_008.fits` - `hdr_047_409_416.fits`
  - 056: `hdr_056_201_208.fits` - `hdr_056_281_288.fits`

Site 015 cannot be processed.


## 2. Command-line options

`catetool` is a command-line program accepting the following options (arguments):

```
--mode <mode>
```

Mode of operation. Possible values: `single-site` (align HDR images from a single site), `multi-site` (align averaged HDR images of multiple sites), `precalc` (produce aligned images according to precalculated translations and rotations).

Mode `precalc` requires the usage of `--input_list` option. The list file may contain any text in the beginning, followed by:
```
BEGIN LIST
<file1>
<x1> <y1> <angle> <scale>
<file2>
<x2> <y2> <angle> <scale>
...
```
The `angle` (deg.) applies with (x, y) as the rotation center and may be skipped (will be presumed 0.0). Likewise, `scale` uses (x, y) as the scaling center and may be skipped (will be presumed 1.0).
The x, y must be specified at least for the first and the last file. Any values not specified will be linearly interpolated.

```
--input_files <file1 file2 ...>
```

Input files in alignment order.

```
--input_list <file>
```

File containing the list of input files in alignment order (one file per line). If used with `--mode precalc`, the format is more complex (see `--mode`).

```
--output_dir <directory>
```

Output directory in which to save the aligned images and reference blocks.

```
--ref_pos <x> <y>
```

Position of the center of the big prominence's base in the first input file. Required for `--mode single-site`.

```
--output_avg_file <file>
```

Path and name of the averaged output file. Default: `averaged.fits`.

```
--save_aligned <yes|no>
```

Whether to save aligned files in `single-site` or `precalc` mode. Default: `no` for `single-site`, `yes` for `precalc`.

```
--exclude_moon_diameter <moon diameter in pixels>
```

If specified, pixels covered by the Moon are skipped when creating the averaged image.

```
--background_threshold <value>
```

Specifies the brightness threshold of sky background used with `exclude_moon_diameter`. Default: 3800.0.

```
--log_level <quiet|info|verbose>
```

Chooses the amount of messages to print during processing.


## 3. Scripts

The `*.sh` (for Bash) and `*.bat` (for Windows CMD) scripts simplify the common tasks:

- `align_sites`: intra-site alignment of one or more sites. Set the values in the "USER-CONFIGURABLE PARAMETERS" section as needed, and/or modify the associated `all_sites.txt` file which serves as input. (Refer to in-script comments for details.)

- `align_sites_together`: inter-site alignment for the final time lapse. Set the values in the "USER-CONFIGURABLE PARAMETERS" section as needed. (Refer to in-script comments for details.)

The `*.sh` scripts can be run from command line under Linux, OS X (and similar systems) and under MS Windows in MSYS2 shell. The `*.bat` scripts are meant to be run directly from Windows Explorer or from MS Windows CMD shell, when using the MS Windows binary distribution of `catetool` (`catetool-win64.zip`).


## 4. Building from source code

Building from sources requires the Rust language toolchain and the CFITSIO (development) library.

Note that the initial build will take longer, as all the dependencies have to be downloaded and built first.


### 4.1. Linux, OS X (and other Unices)

Go to https://www.rust-lang.org/tools/install and follow the default instructions to install Rust.

Installation of CFITSIO depends on the operating system; e.g., on Fedora run:
```bash
$ sudo dnf install cfitsio-devel
```
On Ubuntu:
```
$ sudo apt install libcfitsio-dev
```

To build `catetool`, go to the sources directory and execute:
```
$ cargo build --release
```

Afterwards, the program can be run (from the sources directory) with:
```
$ target/release/catetool <options...>
```
and the scripts as follows:
```
$ scripts/align_sites.sh
```


### 4.2. MS Windows

Building under MS Windows has been tested in MSYS2 environment and the GNU variant of the Rust toolchain.

To install MSYS2, go to https://www.msys2.org/ and follow the installation instructions. Next, open the MSYS2/MinGW64 shell (by default, via `c:\msys64\mingw64.exe`) and install the base development toolchain and additional libraries:
```
$ pacman -S base-devel mingw64/mingw-w64-x86_64-pkg-config mingw64/mingw-w64-x86_64-cfitsio
```

To install Rust, go to https://forge.rust-lang.org/infra/other-installation-methods.html and download and run the `x86_64-pc-windows-gnu` installer. After the installation, in the MSYS2/MinGW64 shell execute:
```
$ export PATH=/c/Users/MYUSER/.cargo/bin:$PATH
```

Change to the `catetool` sources directory and build it with:
```
$ cargo build --release
```

Afterwards, the program can be run (from the sources directory) with:
```
$ target/release/catetool <options...>
```
and the scripts as follows:
```
$ scripts/align_sites.sh
```


# 5. Change log

**0.1.3** (2020-08-15)

  - Additional command-line options to enable alignment of low-brightness sites.

**0.1.2** (2020-06-14)

  - Added support for --input_list.

**0.1.1** (2020-05-24)

  - Skipping Moon-obscured pixels during averaging in single-site mode.
  - Image scale change compensation in multi-site mode.

**0.1** (2020-05-08)

  - Initial revision.
