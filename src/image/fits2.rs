//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! FITS image format handling code (low-level).
//!

use super::{Image, Palette, PixelFormat};
use std::mem::MaybeUninit;
use fitsio_sys;

//TODO: handle other pixel formats

const TFLOAT: std::os::raw::c_int = 42;
const FLOAT_IMG: std::os::raw::c_int = -32;

#[derive(Debug)]
pub enum FitsError {
    CannotOpenFile,
    InvalidImage,
    UnsupportedPixelFormat,
    NoImageInFile,
    CannotReadImage,
    CannotCreateFile,
    CannotCreateImage,
    CannotWriteImage,
    CannotReadKeyword
}

struct FitsFileHandle {
    fptr: *mut fitsio_sys::fitsfile
}

impl Drop for FitsFileHandle {
    fn drop(&mut self) {
        let mut status = 0;
        unsafe { fitsio_sys::ffclos(self.fptr, &mut status) };
    }
}

/// Returns (file, width, height).
fn load_fits_file(file_name: &str) -> Result<(FitsFileHandle, u32, u32), FitsError> {
    let file_name_image0 = file_name.to_string() + "[0]";
    let mut file = FitsFileHandle{ fptr: std::ptr::null_mut() };
    let mut status = 0;
    const READONLY: std::os::raw::c_int = 0;
    unsafe { fitsio_sys::ffopen(
        &mut file.fptr as *mut *mut _,
        std::ffi::CString::new(file_name_image0).unwrap().as_ptr(),
        READONLY,
        &mut status
    ) };
    if status != 0 { return Err(FitsError::CannotOpenFile); }

    let mut bits_per_pixel = MaybeUninit::<std::os::raw::c_int>::uninit();
    let mut num_axes = MaybeUninit::<std::os::raw::c_int>::uninit();
    let mut dimensions: [std::os::raw::c_long; 3] = [0; 3];
    unsafe { fitsio_sys::ffghpr(
        file.fptr as *mut _,
        3,
        std::ptr::null_mut(),
        bits_per_pixel.as_mut_ptr(),
        num_axes.as_mut_ptr(),
        dimensions.as_mut_ptr(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        &mut status
    ) };
    if status != 0 {
        return Err(FitsError::InvalidImage);
    }
    let num_axes = unsafe { num_axes.assume_init() };
    let bits_per_pixel = unsafe { bits_per_pixel.assume_init() };

    if num_axes != 2 || dimensions[0] <= 0 || dimensions[1] <= 0 {
        return Err(FitsError::InvalidImage);
    }

    if bits_per_pixel != FLOAT_IMG {
        return Err(FitsError::UnsupportedPixelFormat);
    }

    Ok((file, dimensions[0] as u32, dimensions[1] as u32))
}

pub fn load_fits(file_name: &str) -> Result<Image, FitsError> {
    let (file, width, height) = load_fits_file(file_name)?;

    let mut image = Image::new(width, height, None, PixelFormat::Mono32f, None, true);

    let mut status = 0;

    // FITS rows are stored in reverse order
    for y in 0..height {
        unsafe { fitsio_sys::ffgpv(
            file.fptr as *mut _,
            TFLOAT,
            (1 + y * width) as _,
            width as _,
            std::ptr::null_mut(),
            image.line_mut::<f32>(height - 1 - y) as *mut _ as *mut _,
            std::ptr::null_mut(),
            &mut status
        ) };
        if status != 0 {
            return Err(FitsError::CannotReadImage);
        }
    }

    Ok(image)
}

/// Returns metadata (width, height, ...) without reading the pixel data.
pub fn fits_metadata(file_name: &str) -> Result<(u32, u32, PixelFormat, Option<Palette>), FitsError> {
    let (_, width, height) = load_fits_file(file_name)?;

    Ok((width, height, PixelFormat::Mono32f, None))
}

pub fn save_fits(image: &Image, file_name: &str) -> Result<(), FitsError> {
    assert!(image.pixel_format() == PixelFormat::Mono32f);

    let mut dimensions: [std::os::raw::c_long; 2] = [
        image.width() as _,
        image.height() as _
    ];

    let mut status = 0;
    let mut file = FitsFileHandle{ fptr: std::ptr::null_mut() };
    // a leading "!" overwrites an existing file
    unsafe { fitsio_sys::ffinit(
        &mut file.fptr as *mut *mut _,
        std::ffi::CString::new("!".to_string() + file_name).unwrap().as_ptr(),
        &mut status
    ) };
    if status != 0 { return Err(FitsError::CannotCreateFile); }

    unsafe { fitsio_sys::ffcrim(file.fptr as *mut _, FLOAT_IMG, 2, dimensions.as_mut_ptr(), &mut status) };
    if status != 0 { return Err(FitsError::CannotCreateImage); }

    // FITS rows are stored in reverse order
    for y in 0..image.height() {
        unsafe { fitsio_sys::ffppr(
            file.fptr as *mut _,
            TFLOAT,
            (1 + (image.height() - 1 - y) * image.width()) as _,
            image.width() as _,
            image.line_raw(y).as_ptr() as *mut _,
            &mut status
        ) };
        if status != 0 { return Err(FitsError::CannotWriteImage); }
    }

    Ok(())
}

pub fn get_fits_keywords(file_name: &str, keywords: &[&str]) -> Result<Vec<String>, FitsError> {
    let (fits_file, _, _) = load_fits_file(file_name)?;

    let mut result = vec![];

    for keyword in keywords {
        let mut key_value_ptr = MaybeUninit::<*mut std::os::raw::c_char>::uninit();
        let mut status = 0;
        unsafe { fitsio_sys::ffgkls(
            fits_file.fptr as *mut _,
            std::ffi::CString::new(*keyword).unwrap().as_ptr(),
            &mut key_value_ptr as *mut _ as *mut _,
            std::ptr::null_mut(),
            &mut status
        ); }
        if status != 0 {
            return Err(FitsError::CannotReadKeyword);
        }

        let key_value_ptr = unsafe { key_value_ptr.assume_init() };
        let key_value = String::from(unsafe { std::ffi::CStr::from_ptr(key_value_ptr) }.to_str().unwrap());
        unsafe { fitsio_sys::fffree(key_value_ptr as *mut _, &mut status); }

        result.push(key_value);
    }

    Ok(result)
}