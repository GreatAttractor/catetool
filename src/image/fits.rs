//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! FITS image format handling code.
//!

use super::{Image, Palette, PixelFormat};
use fitsio::{FitsFile};
use fitsio::hdu::{FitsHdu, HduInfo};
use fitsio::images::{ImageDescription, ImageType};

//TODO: handle other pixel formats

macro_rules! get_checked {
    ($e:expr) => {
        match $e {
            Ok(value) => Ok(value),
            _ => Err(FitsError::Unknown)
        }
    }
}

#[derive(Debug)]
pub enum FitsError {
    Internal(fitsio::errors::Error),
    Unknown,
    NoImageInFile
}

impl From<fitsio::errors::Error> for FitsError {
    fn from(err: fitsio::errors::Error) -> FitsError {
        FitsError::Internal(err)
    }
}

pub fn load_fits(file_name: &str) -> Result<Image, FitsError> {
    let (mut fits_file, hdu) = load_primary_hdu(file_name)?;

    let (width, height) = match &hdu.info {
        HduInfo::ImageInfo{ shape, image_type: _ } => (shape[1], shape[0]),
        _ => return Err(FitsError::NoImageInFile)
    };

    let fits_pixels: Vec<f32> = hdu.read_image(&mut fits_file).unwrap();
    assert!(fits_pixels.len() == width*height);

    let mut image = Image::new(width as u32, height as u32, None, PixelFormat::Mono32f, None, false);
    for y in 0..height {
        let row_start = y * width;
        image.line_mut::<f32>(y as u32).copy_from_slice(&fits_pixels[row_start..row_start + width]);
    }

    Ok(image)
}

fn load_primary_hdu(file_name: &str) -> Result<(FitsFile, FitsHdu), FitsError> {
    let mut fits_file = get_checked!(FitsFile::open(file_name))?;
    let hdu = get_checked!(fits_file.primary_hdu())?;
    Ok((fits_file, hdu))
}

/// Returns metadata (width, height, ...) without reading the pixel data.
pub fn fits_metadata(file_name: &str) -> Result<(u32, u32, PixelFormat, Option<Palette>), FitsError> {
    let (_, hdu) = load_primary_hdu(file_name)?;

    let (width, height) = match &hdu.info {
        HduInfo::ImageInfo{ shape, image_type: _ } => (shape[1], shape[0]),
        _ => return Err(FitsError::NoImageInFile)
    };

    Ok((width as u32, height as u32, PixelFormat::Mono32f, None))
}

pub fn save_fits(image: &Image, file_name: &str) -> Result<(), FitsError> {
    assert!(image.pixel_format() == PixelFormat::Mono32f);

    let descr = ImageDescription{
        data_type: ImageType::Float,
        dimensions: &[image.height() as usize, image.width() as usize]
    };

    let _ = std::fs::remove_file(file_name);

    let mut fits_file = FitsFile::create(file_name)
        .with_custom_primary(&descr)
        .open()?;

    let hdu = get_checked!(fits_file.primary_hdu())?;

    for y in 0..image.height() {
        let src_line = image.line::<f32>(y);
        let start_idx = (y * image.width()) as usize;
        hdu.write_section(&mut fits_file, start_idx, start_idx + image.width() as usize, src_line).unwrap();
    }

    Ok(())
}
