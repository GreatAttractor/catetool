//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Low-level utility functions and macros for images.
//!

use super::{Image};
use std;
use std::fs::File;
use std::io::{self, Read, Write};
use std::slice;


/// Rounds `x` up to the closest multiple of `n`.
#[macro_export]
macro_rules! upmult {
    ($x:expr, $n:expr) => { (($x) + ($n) - 1) / ($n) * ($n) }
}


/// Produces a range of specified length.
#[macro_export]
macro_rules! range { ($start:expr, $len:expr) => { $start .. $start + $len } }

pub fn read_struct<T, R: Read>(read: &mut R) -> io::Result<T> {
    let num_bytes = ::std::mem::size_of::<T>();
    unsafe {
        let mut s = std::mem::MaybeUninit::<T>::uninit();
        let buffer: &mut [u8] = slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, num_bytes);
        match read.read_exact(buffer) {
            Ok(()) => Ok(s.assume_init()),
            Err(e) => { ::std::mem::forget(s); Err(e) }
        }
    }
}


pub fn read_vec<T>(file: &mut File, len: usize) -> io::Result<Vec<T>> {
    let mut vec = alloc_uninitialized::<T>(len);
    let num_bytes = ::std::mem::size_of::<T>() * vec.len();
    let buffer = unsafe{ slice::from_raw_parts_mut(vec[..].as_mut_ptr() as *mut u8, num_bytes) };
    file.read_exact(buffer)?;
    Ok(vec)
}


pub fn write_struct<T, W: Write>(obj: &T, write: &mut W) -> Result<(), io::Error> {
    let num_bytes = ::std::mem::size_of::<T>();
    unsafe {
        let buffer = slice::from_raw_parts(obj as *const T as *const u8, num_bytes);
        write.write_all(buffer)
    }
}


/// Allocates an uninitialized `Vec<T>` having `len` elements.
/// FIXME: allow only primitive `T`.
pub fn alloc_uninitialized<T>(len: usize) -> Vec<T> {
    let mut v = Vec::<T>::with_capacity(len);
    unsafe { v.set_len(len); }

    v
}

/// Changes endianess of 16-bit words.
pub fn swap_words16(img: &mut Image) {
    for val in img.pixels_mut::<u16>() {
        *val = u16::swap_bytes(*val);
    }
}

pub fn is_machine_big_endian() -> bool {
    u16::to_be(0x1122u16) == 0x1122u16
}