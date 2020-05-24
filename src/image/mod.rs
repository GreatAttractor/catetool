//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Image data structure, I/O and basic operations.
//!

#[macro_use]
mod utils; // must be first due to exported macros
mod bmp;
//mod fits;
mod fits2; // using this temporarily until fitsio crate is fixed and builds under Windows 64-bit
mod tiff;

pub mod point;

pub use fits2::get_fits_keywords;
pub use point::{Point, Rect};
use std::any::Any;
use std::cmp::{min, max};
use std::convert::From;
use std::default::Default;
use std::path::Path;
use std::ptr;
use std::slice;


#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FileType {
    /// Determined automatically from file name extension
    Auto,
    Bmp,
    Tiff,
    Fits
}


fn file_type_from_ext(file_name: &str) -> FileType {
    match Path::new(file_name).extension() {
        Some(ext) => match ext.to_str().unwrap().to_lowercase().as_str() {
                         "bmp" => FileType::Bmp,
                         "tif" | "tiff" => FileType::Tiff,
                         "fit" | "fits" => FileType::Fits,
                         _ => panic!(format!("Unrecognized file extension: {}", ext.to_str().unwrap()))
                     },
        _ => panic!(format!("No file extension in file name: {}", file_name))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    /// 8 bits per pixel, values from a 256-entry palette.
    Pal8,
    Mono8,
    /// LSB = R, MSB = B.
    RGB8,
    /// LSB = B, MSB = R.
    BGR8,
    /// LSB = B, MSB = A or unused.
    BGRA8,

    Mono16,
    RGB16,
    RGBA16,

    Mono32f,
    RGB32f,

    Mono64f,
    RGB64f,

    CfaRGGB8,
    CfaGRBG8,
    CfaGBRG8,
    CfaBGGR8,

    CfaRGGB16,
    CfaGRBG16,
    CfaGBRG16,
    CfaBGGR16
}


impl PixelFormat {
    /// Returns true if this is a Color Filter Array pixel format.
    pub fn is_cfa(&self) -> bool {
        [PixelFormat::CfaRGGB8,
         PixelFormat::CfaGRBG8,
         PixelFormat::CfaGBRG8,
         PixelFormat::CfaBGGR8,
         PixelFormat::CfaRGGB16,
         PixelFormat::CfaGRBG16,
         PixelFormat::CfaGBRG16,
         PixelFormat::CfaBGGR16].contains(&self)
    }
}


/// Demosaicing (debayering) method using when converting CFA images to Mono/RGB.
pub enum DemosaicMethod {
    /// Fast, but low-quality.
    ///
    /// Used internally during image alignment, quality estimation and ref. point alignment.
    ///
    Simple,

    /// High-quality and slower; used internally during stacking phase.
    HqLinear,
}

pub fn get_matching_mono_format<T: Default + Any>() -> PixelFormat {
    let t = &T::default() as &dyn Any;
    if t.is::<u8>() {
        PixelFormat::Mono8
    } else if t.is::<u16>() {
        PixelFormat::Mono16
    } else if t.is::<f32>() {
        PixelFormat::Mono32f
    } else if t.is::<f64>() {
        PixelFormat::Mono64f
    } else {
        panic!("Unsupported type.")
    }
}

/// Asserts that `T` is the type of pixel values (in each channel) corresponding to `pix_fmt`.
fn verify_pix_type<T: Default + Any>(pix_fmt: PixelFormat) {
    let t = &T::default() as &dyn Any;
    match pix_fmt {
        PixelFormat::Pal8     |
        PixelFormat::Mono8    |
        PixelFormat::RGB8     |
        PixelFormat::BGR8     |
        PixelFormat::BGRA8    |
        PixelFormat::CfaRGGB8 |
        PixelFormat::CfaGRBG8 |
        PixelFormat::CfaGBRG8 |
        PixelFormat::CfaBGGR8 => assert!(t.is::<u8>()),

        PixelFormat::Mono16    |
        PixelFormat::RGB16     |
        PixelFormat::RGBA16    |
        PixelFormat::CfaRGGB16 |
        PixelFormat::CfaGRBG16 |
        PixelFormat::CfaGBRG16 |
        PixelFormat::CfaBGGR16 => assert!(t.is::<u16>()),

        PixelFormat::Mono32f | PixelFormat::RGB32f => assert!(t.is::<f32>()),

        PixelFormat::Mono64f | PixelFormat::RGB64f => assert!(t.is::<f64>()),
    }}


pub fn num_channels(pix_fmt: PixelFormat) -> usize {
    match pix_fmt {
        PixelFormat::Pal8      |
        PixelFormat::Mono8     |
        PixelFormat::Mono16    |
        PixelFormat::Mono32f   |
        PixelFormat::Mono64f   |
        PixelFormat::CfaRGGB8  |
        PixelFormat::CfaGRBG8  |
        PixelFormat::CfaGBRG8  |
        PixelFormat::CfaBGGR8  |
        PixelFormat::CfaRGGB16 |
        PixelFormat::CfaGRBG16 |
        PixelFormat::CfaGBRG16 |
        PixelFormat::CfaBGGR16 => 1,

        PixelFormat::RGB8   |
        PixelFormat::BGR8   |
        PixelFormat::RGB16  |
        PixelFormat::RGB32f |
        PixelFormat::RGB64f => 3,

        PixelFormat::BGRA8 |
        PixelFormat::RGBA16 => 4
    }
}

pub fn bytes_per_pixel(pix_fmt: PixelFormat) -> usize {
    match pix_fmt {
        PixelFormat::Pal8 | PixelFormat::Mono8 => 1,
        PixelFormat::RGB8 | PixelFormat::BGR8  => 3,
        PixelFormat::BGRA8   => 4,
        PixelFormat::Mono16  => 2,
        PixelFormat::RGB16   => 6,
        PixelFormat::RGBA16  => 8,
        PixelFormat::Mono32f => 4,
        PixelFormat::RGB32f  => 16,
        PixelFormat::Mono64f => 8,
        PixelFormat::RGB64f  => 24,

        _ => panic!()
    }
}


pub fn bytes_per_channel(pix_fmt: PixelFormat) -> usize {
    match pix_fmt {
        PixelFormat::Pal8     |
        PixelFormat::Mono8    |
        PixelFormat::RGB8     |
        PixelFormat::BGR8     |
        PixelFormat::BGRA8    |
        PixelFormat::CfaRGGB8 |
        PixelFormat::CfaGRBG8 |
        PixelFormat::CfaGBRG8 |
        PixelFormat::CfaBGGR8 => 1,

        PixelFormat::Mono16    |
        PixelFormat::RGB16     |
        PixelFormat::RGBA16    |
        PixelFormat::CfaRGGB16 |
        PixelFormat::CfaGRBG16 |
        PixelFormat::CfaGBRG16 |
        PixelFormat::CfaBGGR16 => 2,

        PixelFormat::Mono32f | PixelFormat::RGB32f => 4,

        PixelFormat::Mono64f | PixelFormat::RGB64f => 8,
    }
}

#[derive(Copy)]
pub struct Palette {
    pub pal: [u8; 3 * Palette::NUM_ENTRIES]
}

impl std::fmt::Debug for Palette {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Palette [{}, {}, {}, ...]", self.pal[0], self.pal[1], self.pal[2])
    }
}


impl Palette {
    pub const NUM_ENTRIES: usize = 256;
}


impl Clone for Palette {
    fn clone(&self) -> Palette { *self }
}


impl Default for Palette {
    fn default() -> Palette { Palette{ pal: [0; 3 * Palette::NUM_ENTRIES] }}
}

pub struct Image {
    width: u32,
    height: u32,
    pix_fmt: PixelFormat,
    palette: Option<Palette>,
    pixels: Vec<u8>,
    bytes_per_line: usize
}

#[derive(Debug)]
pub enum ImageError {
    BmpError(bmp::BmpError),
    TiffError(tiff::TiffError),
    FitsError(fits2::FitsError)
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Image {}x{}, {:?}, bytes_per_line = {}, pixels = {:?}...",
            self.width,
            self.height,
            self.pix_fmt,
            self.bytes_per_line,
            &self.pixels[..8]
        )
    }
}

impl Image {
    /// Adds `margin`-sized borders on each side.
    pub fn inflate(&self, margin: u32) -> Image {
        let mut result = Image::new(
            self.width + 2 * margin,
            self.height + 2 * margin,
            None,
            self.pix_fmt,
            self.palette,
            true
        );
        self.resize_and_translate_into(
            &mut result,
            Point::zero(),
            self.width,
            self.height,
            Point{ x: margin as i32, y: margin as i32 },
            true
        );

        result
    }

    /// Removes `margin`-sized borders on each side.
    pub fn deflate(&self, margin: u32) -> Image {
        assert!(self.width > 2 * margin);
        assert!(self.height > 2 * margin);

        let mut result = Image::new(
            self.width - 2 * margin,
            self.height - 2 * margin,
            None,
            self.pix_fmt,
            self.palette,
            true
        );
        self.resize_and_translate_into(
            &mut result,
            Point{ x: margin as i32, y: margin as i32 },
            self.width - 2 * margin,
            self.height - 2 * margin,
            Point::zero(),
            true
        );

        result
    }


    pub fn take_pixel_data(self) -> Vec<u8> {
        self.pixels
    }

    pub fn load(file_name: &str, file_type: FileType) -> Result<Image, ImageError> {
        let ftype = if file_type == FileType::Auto { file_type_from_ext(file_name) } else { file_type };
        match ftype {
            FileType::Bmp => bmp::load_bmp(file_name).map_err(ImageError::BmpError),
            FileType::Tiff => tiff::load_tiff(file_name).map_err(ImageError::TiffError),
            FileType::Fits => fits2::load_fits(file_name).map_err(ImageError::FitsError),
            FileType::Auto => unreachable!()
        }
    }

    /// Overwrites existing file.
    pub fn save(&self, file_name: &str, file_type: FileType) -> Result<(), ImageError> {
        let ftype = if file_type == FileType::Auto { file_type_from_ext(file_name) } else { file_type };
        match ftype {
            FileType::Bmp => bmp::save_bmp(&self, file_name).map_err(ImageError::BmpError),
            FileType::Tiff => tiff::save_tiff(&self, file_name).map_err(ImageError::TiffError),
            FileType::Fits => fits2::save_fits(&self, file_name).map_err(ImageError::FitsError),
            FileType::Auto => unreachable!()
        }
    }

    /// Returns width, height, pixel format, palette.
    pub fn image_metadata(file_name: &str, file_type: FileType) -> Result<(u32, u32, PixelFormat, Option<Palette>), ImageError> {
        let ftype = if file_type == FileType::Auto { file_type_from_ext(file_name) } else { file_type };
        match ftype {
            FileType::Bmp => bmp::bmp_metadata(file_name).map_err(ImageError::BmpError),
            FileType::Tiff => tiff::tiff_metadata(file_name).map_err(ImageError::TiffError),
            FileType::Fits => fits2::fits_metadata(file_name).map_err(ImageError::FitsError),
            FileType::Auto => unreachable!()
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.pix_fmt
    }

    /// Creates a new image using the specified storage.
    ///
    /// `pixels` must have enough space. `palette` is used only if `pix_fmt` equals `Pal8`.
    ///
    pub fn new_from_pixels(
        width: u32,
        height: u32,
        mut bytes_per_line: Option<usize>,
        pix_fmt: PixelFormat,
        pal: Option<Palette>,
        pixels: Vec<u8>
    ) -> Image {
        match bytes_per_line {
            Some(num) => {
                assert!(num as usize >= width as usize * bytes_per_pixel(pix_fmt));
                assert!((num - width as usize * bytes_per_pixel(pix_fmt)) % bytes_per_pixel(pix_fmt) == 0);
            },
            None => bytes_per_line = Some(width as usize * bytes_per_pixel(pix_fmt))
        }

        assert!(pixels.len() >= height as usize * bytes_per_line.unwrap());

        Image{
            width: width,
            height: height,
            pix_fmt: pix_fmt,
            palette: pal,
            pixels: pixels,
            bytes_per_line: bytes_per_line.unwrap()
        }
    }

    /// Creates a new image.
    ///
    /// `palette` is used only if `pix_fmt` equals `Pal8`.
    ///
    pub fn new(width: u32, height: u32, mut bytes_per_line: Option<usize>, pix_fmt: PixelFormat, palette: Option<Palette>, zero_fill: bool) -> Image {
        match bytes_per_line {
            Some(num) => assert!(num as usize >= width as usize * bytes_per_pixel(pix_fmt)),
            None => bytes_per_line = Some(width as usize * bytes_per_pixel(pix_fmt))
        }

        let pixels: Vec<u8>;
        let byte_count = height as usize * bytes_per_line.unwrap();
        if zero_fill {
            pixels = vec![0; byte_count];
        } else {
            pixels = utils::alloc_uninitialized(byte_count);
        }

        Image::new_from_pixels(width, height, bytes_per_line, pix_fmt, palette, pixels)
    }

    pub fn bytes_per_line(&self) -> usize {
        self.bytes_per_line
    }

    /// Returns channel values per line (including padding, if any).
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn values_per_line<T: Any + Default>(&self) -> usize {
        verify_pix_type::<T>(self.pix_fmt);
        self.bytes_per_line / std::mem::size_of::<T>()
    }

    /// Returns pixels (including row padding, if any).
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn pixels<T: Any + Default>(&self) -> &[T] {
        verify_pix_type::<T>(self.pix_fmt);

        let ptr: *const u8 = self.pixels[..].as_ptr();
        unsafe {
            slice::from_raw_parts(
                ptr as *const T,
                self.bytes_per_line / std::mem::size_of::<T>() * self.height as usize
            )
        }
    }


    /// Returns mutable pixels (including row padding, if any).
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn pixels_mut<T: Any + Default>(&mut self) -> &mut [T] {
        verify_pix_type::<T>(self.pix_fmt);

        let ptr: *mut u8 = self.pixels[..].as_mut_ptr();
        unsafe {
            slice::from_raw_parts_mut(
                ptr as *mut T,
                self.bytes_per_line / std::mem::size_of::<T>() * self.height as usize
            )
        }
    }

    /// For a Mono8 image, returns pixels starting from `start` coordinates.
    pub fn mono8_pixels_from(&self, start: Point) -> &[u8] {
        assert!(self.pix_fmt == PixelFormat::Mono8);
        &self.pixels[(start.y as usize) * self.bytes_per_line + start.x as usize ..]
    }

    /// Returns all pixels as raw bytes (regardless of pixel format).
    pub fn raw_pixels(&self) -> &[u8] {
        &self.pixels[..]
    }

    /// Returns all pixels as raw bytes (regardless of pixel format).
    pub fn raw_pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels[..]
    }

    /// Returns a line as raw bytes (regardless of pixel format).
    pub fn line_raw(&self, y: u32) -> &[u8] {
        &self.pixels[range!(y as usize * self.bytes_per_line, self.bytes_per_line)]
    }

    /// Returns a mutable line as raw bytes (regardless of pixel format).
    pub fn line_raw_mut(&mut self, y: u32) -> &mut [u8] {
        &mut self.pixels[range!(y as usize * self.bytes_per_line, self.bytes_per_line)]
    }

    /// Returns image line.
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn line<T: Any + Default>(&self, y: u32) -> &[T] {
        assert!(y < self.height);
        let vals_per_line = self.values_per_line::<T>();

        &self.pixels::<T>()[range!(y as usize * vals_per_line, vals_per_line)]
    }

    /// Returns mutable image line.
    ///
    /// `T` must correspond to the image's pixel format.
    ///
    pub fn line_mut<T: Any + Default>(&mut self, y: u32) -> &mut [T] {
        assert!(y < self.height);
        let vals_per_line = self.values_per_line::<T>();

        &mut self.pixels_mut::<T>()[range!(y as usize * vals_per_line, vals_per_line)]
    }

    pub fn palette(&self) -> &Option<Palette> {
        &self.palette
    }

    pub fn palette_mut(&mut self) -> &mut Option<Palette> {
        &mut self.palette
    }

    pub fn img_rect(&self) -> Rect {
        Rect{ x: 0, y: 0, width: self.width as u32, height: self.height as u32 }
    }

    /// Calculates and returns image moments: M00, M10, M01. `T`: type of pixel values.
    ///
    /// `pixels` must not contain palette entries. To use the function with a palettized image,
    /// convert it to other format first.
    ///
    fn moments<T: Any + Default>(&self, img_fragment: Rect) -> (f64, f64, f64)
        where T: Copy, f64: From<T> {

        let mut m00: f64 = 0.0; // image moment 00, i.e. sum of pixels' brightness
        let mut m10: f64 = 0.0; // image moment 10
        let mut m01: f64 = 0.0; // image moment 01

        let num_channels = num_channels(self.pix_fmt);

        for y in range!(img_fragment.y, img_fragment.height as i32) {
            let line = self.line::<T>(y as u32);
            for x in range!(img_fragment.x, img_fragment.width as i32) {
                let mut current_brightness: f64 = 0.0;

                for i in 0..num_channels {
                    current_brightness += f64::from(line[x as usize + i]);
                }

                m00 += current_brightness;
                m10 += (x - img_fragment.x) as f64 * current_brightness;
                m01 += (y - img_fragment.y) as f64 * current_brightness;
            }
        }

        (m00, m10, m01)
    }


    /// Finds the centroid of the specified image fragment.
    ///
    /// Returned coords are relative to `img_fragment`. Image must not be `PixelFormat::Pal8`.
    /// If `img_fragment` is `None`, whole image is used.
    ///
    pub fn centroid(&self, img_fragment: Option<Rect>) -> Point {
        let m00: f64;
        let m10: f64;
        let m01: f64;

        let img_fragment = match img_fragment {
            Some(r) => r,
            None => self.img_rect()
        };

        match self.pix_fmt {
            PixelFormat::Pal8     |
            PixelFormat::Mono8    |
            PixelFormat::RGB8     |
            PixelFormat::BGR8     |
            PixelFormat::BGRA8    |
            PixelFormat::CfaRGGB8 |
            PixelFormat::CfaGRBG8 |
            PixelFormat::CfaGBRG8 |
            PixelFormat::CfaBGGR8 => {
                let moments = self.moments::<u8>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            },

            PixelFormat::Mono16    |
            PixelFormat::RGB16     |
            PixelFormat::RGBA16    |
            PixelFormat::CfaRGGB16 |
            PixelFormat::CfaGRBG16 |
            PixelFormat::CfaGBRG16 |
            PixelFormat::CfaBGGR16 => {
                let moments = self.moments::<u16>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            },

            PixelFormat::Mono32f | PixelFormat::RGB32f => {
                let moments = self.moments::<f32>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            },

            PixelFormat::Mono64f | PixelFormat::RGB64f => {
                let moments = self.moments::<f64>(img_fragment);
                m00 = moments.0;
                m10 = moments.1;
                m01 = moments.2;
            }
        }

        if m00 == 0.0 {
            return Point{ x: img_fragment.width as i32 / 2, y: img_fragment.height as i32 / 2 };
        } else {
            return Point{ x: (m10/m00) as i32, y: (m01/m00) as i32 };
        }
    }


    /// Converts a fragment of the image to `dest_img`'s pixel format and writes it into `dest_img`.
    ///
    /// The fragment to convert starts at `src_pos` in `&self`, has `width`x`height` pixels and will
    /// be written to `dest_img` starting at `dest_pos`. Cropping is performed if necessary.
    /// If `&self` is in raw color format, the CFA pattern will be appropriately adjusted,
    /// depending on `src_pos` and `dest_pos`.
    ///
    pub fn convert_pix_fmt_of_subimage_into(&self,
                                            dest_img: &mut Image,
                                            src_pos: Point,
                                            dest_pos: Point,
                                            width: u32,
                                            height: u32,
                                            demosaic_method: Option<DemosaicMethod>) {

        // Converting to Pal8 or raw color (CFA) is not supported
        assert!(!(dest_img.pix_fmt == PixelFormat::Pal8 && self.pix_fmt != PixelFormat::Pal8));
        assert!(!dest_img.pix_fmt.is_cfa());

        // Source position cropped so that the source rectangle fits in `src_img`
        let mut actual_src_pos = Point{ x: max(0, src_pos.x),
                                        y: max(0, src_pos.y) };

        let mut actual_width = min(width as i32, self.width as i32 - actual_src_pos.x) as usize;
        let mut actual_height = min(height as i32, self.width as i32 - actual_src_pos.y) as usize;

        // Destination position based on `src_pos` and further cropped so that the dest. rectangle fits in `dest_img`
        let mut actual_dest_pos = Point{ x: dest_pos.x + (src_pos.x - actual_src_pos.x),
                                         y: dest_pos.y + (src_pos.y - actual_src_pos.y) };

        if actual_dest_pos.x >= dest_img.width as i32 ||
           actual_dest_pos.y >= dest_img.height as i32 ||
           actual_src_pos.x >= self.width as i32 ||
           actual_src_pos.y >= self.height as i32 {

            return;
        }

        actual_dest_pos.x = max(0, actual_dest_pos.x);
        actual_dest_pos.y = max(0, actual_dest_pos.y);

        actual_width = min(actual_width as i32, dest_img.width as i32 - actual_dest_pos.x) as usize;
        actual_height = min(actual_height as i32, dest_img.height as i32 - actual_dest_pos.y) as usize;

        // Reflect in the source rectangle any cropping imposed by `dest_img`
        actual_src_pos.x += actual_dest_pos.x - dest_pos.x;
        actual_src_pos.y += actual_dest_pos.y - dest_pos.y;

        if self.pix_fmt == dest_img.pix_fmt {
            // No conversion required, just copy the data

            let bpp = bytes_per_pixel(self.pix_fmt);
            let copy_line_len = actual_width * bpp;

            let mut src_ofs = actual_src_pos.y as usize * self.bytes_per_line;
            let mut dest_ofs = actual_dest_pos.y as usize * dest_img.bytes_per_line;

            for _ in 0..actual_height {
                let copy_to = dest_ofs + actual_dest_pos.x as usize * bpp;
                let copy_from = src_ofs + actual_src_pos.x as usize * bpp;

                dest_img.pixels[range!(copy_to, copy_line_len)]
                    .copy_from_slice(&self.pixels[range!(copy_from, copy_line_len)]);

                src_ofs += self.bytes_per_line;
                dest_ofs += dest_img.bytes_per_line;
            }

            return;
        }

        let src_step = bytes_per_pixel(self.pix_fmt);
        let dest_step = bytes_per_pixel(dest_img.pix_fmt);

        for y in 0..actual_height {
            let mut src_ofs = (y + actual_src_pos.y as usize) * self.bytes_per_line +
                actual_src_pos.x as usize * bytes_per_pixel(self.pix_fmt);
            let mut dest_ofs = (y + actual_dest_pos.y as usize) * dest_img.bytes_per_line +
                actual_dest_pos.x as usize * bytes_per_pixel(dest_img.pix_fmt);

            /// Returns a slice of `dest_img`'s pixel values of type `T`, beginning at byte offset `dest_ofs`.
            macro_rules! dest { ($len:expr, $T:ty) => { unsafe { slice::from_raw_parts_mut(dest_img.pixels[dest_ofs..].as_mut_ptr() as *mut $T, $len) } }};

            /// Executes the code in block `b` in a loop encompassing a whole line of destination area in `dest_img`.
            macro_rules! convert_whole_line {
                ($b:block) => {
                    for _ in 0..actual_width {
                        $b

                        src_ofs += src_step;
                        dest_ofs += dest_step;
                    }
                }
            };

            match self.pix_fmt {
                PixelFormat::Mono8 => {
                    match dest_img.pix_fmt {
                        PixelFormat::Mono16 =>
                            convert_whole_line!({ dest!(1, u16)[0] = (self.pixels[src_ofs] as u16) << 8 }),

                        PixelFormat::Mono32f =>
                            convert_whole_line!({ dest!(1, f32)[0] = self.pixels[src_ofs] as f32 / 0xFF as f32; }),

                        PixelFormat::Mono64f =>
                            convert_whole_line!({ dest!(1, f64)[0] = self.pixels[src_ofs] as f64 / f64::from(0xFF); }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!({ for i in dest!(3, f32) { *i = self.pixels[src_ofs] as f32 / 0xFF as f32; } }),

                        PixelFormat::RGB64f =>
                            convert_whole_line!({ for i in dest!(3, f64) { *i = self.pixels[src_ofs] as f64 * 1.0 / 0xFF as f64; } }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = self.pixels[src_ofs]; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = self.pixels[src_ofs]; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!({ for i in dest!(3, u16) { *i = (self.pixels[src_ofs] as u16) << 8; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::Mono16 => {
                    /// Returns the current source pixel value as `u16`.
                    macro_rules! src { () => { unsafe { *(self.pixels[src_ofs..].as_ptr() as *const u16) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!() >> 8) as u8; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = src!() as f32 / 0xFFFF as f32; }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!({ for i in dest!(3, f32) { *i = src!() as f32 / 0xFFFF as f32; } }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!() >> 8) as u8; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = (src!() >> 8) as u8; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!({ for i in dest!(3, u16) { *i = src!(); } }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = src!() as f64 / 0xFFFF as f64; }),

                        PixelFormat::RGB64f => convert_whole_line!({ for i in dest!(3, f64) { *i = src!() as f64 / 0xFFFF as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::Mono32f => {
                    /// Returns the current source pixel value as `f32`.
                    macro_rules! src { () => { unsafe { *(self.pixels[src_ofs..].as_ptr() as *const f32) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!() * 0xFF as f32) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!() * 0xFFFF as f32) as u16; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u8);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!() * 0xFF as f32) as u8; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = (src!() * 0xFF as f32) as u8; } }),

                        PixelFormat::RGB16 => convert_whole_line!({ for i in dest!(3, u16) { *i = (src!() * 0xFFFF as f32) as u16; } }),

                        PixelFormat::RGB32f => convert_whole_line!({ for i in dest!(3, f32) { *i = src!(); } }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = src!() as f64; }),

                        PixelFormat::RGB64f => convert_whole_line!({ for i in dest!(3, f64) { *i = src!() as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::Mono64f => {
                    /// Returns the current source pixel value as `f64`.
                    macro_rules! src { () => { unsafe { *(self.pixels[src_ofs..].as_ptr() as *const f64) } }};

                    match dest_img.pix_fmt {

                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!() * 0xFF as f64) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!() * 0xFFFF as f64) as u16; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u8);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!() * 0xFF as f64) as u8; }
                            }),

                        PixelFormat::RGB8 | PixelFormat::BGR8 =>
                            convert_whole_line!({ for i in dest!(3, u8) { *i = (src!() * 0xFF as f64) as u8; } }),

                        PixelFormat::RGB16 => convert_whole_line!({ for i in dest!(3, u16) { *i = (src!() * 0xFFFF as f64) as u16; } }),

                        PixelFormat::RGB32f => convert_whole_line!({ for i in dest!(3, f32) { *i = src!() as f32; } }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = src!() as f32; }),

                        PixelFormat::RGB64f => convert_whole_line!({ for i in dest!(3, f64) { *i = src!(); } }),

                        _ => panic!()
                    }
                },

                // When converting from a color format to mono, use sum (scaled) of all channels as the pixel brightness.

                PixelFormat::Pal8 => {
                    let pal: &Palette = self.palette.iter().next().unwrap();

                    /// Returns the current source pixel converted to RGB (8-bit). Parameter `x` is from 0..3.
                    macro_rules! src { ($x:expr) => { pal.pal[(3 * self.pixels[src_ofs]) as usize + $x] } };

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = ((src!(0) as u32 + src!(1) as u32 + src!(2) as u32) / 3) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (((src!(0) as u32 + src!(1) as u32 + src!(2) as u32) / 3) << 8) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!(0) as u32 + src!(1) as u32 + src!(2) as u32) as f32 / (3.0 * 0xFF as f32); }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = (src!(0) as u32 + src!(1) as u32 + src!(2) as u32) as f64 / (3.0 * 0xFF as f64); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = src!(2-i); }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = src!(2-i); }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u8)[i] = src!(i); } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!(i) as u16) << 8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = (src!(i) as f32) / 0xFF as f32; } }),

                        PixelFormat::RGB64f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f64)[i] = (src!(i) as f64) / 0xFF as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::RGB8 => {
                    /// Returns the current source pixel as `u8` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const u8, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!()[0] + src!()[1] + src!()[2]) as u16 / 3 * 0xFF; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) as f32 / (3.0 * 0xFF as f32); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = src!()[2-i]; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = src!()[2-i]; }
                            }),

                        PixelFormat::RGBA16 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u16);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!()[i] as u16) << 8; }
                            }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] as u16) << 8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32 / 0xFF as f32; } }),


                        _ => panic!()
                    }
                },

                PixelFormat::BGR8 => {
                    /// Returns the current source pixel as `u8` BGR values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const u8, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = (src!()[0] + src!()[1] + src!()[2]) as u16 / 3 * 0xFF; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) as f32 / (3.0 * 0xFF as f32); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = src!()[2-i]; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!({
                                let rgb = dest!(3, u8);
                                for i in 0..3 { rgb[i] = src!()[2-i]; }
                            }),

                        PixelFormat::RGBA16 =>
                            convert_whole_line!({
                                let rgba = dest!(4, u16);
                                rgba[3] = 0xFF;
                                for i in 0..3 { rgba[i] = (src!()[i] as u16) << 8; }
                            }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] as u16) << 8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32 / 0xFF as f32; } }),


                        _ => panic!()
                    }
                },

                PixelFormat::RGB16 => {
                    /// Returns the current source pixel as `u16` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const u16, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = (((src!()[0] as u32 + src!()[1] as u32 + src!()[2] as u32) / 3) >> 8) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = ((src!()[0] as u32 + src!()[1] as u32 + src!()[2] as u32) / 3) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) as f32 / (3.0 * 0xFFFF as f32); }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!()[2-i] >> 8) as u8; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = (src!()[2-i] >> 8) as u8; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u8)[i] = (src!()[i] >> 8) as u8; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32 / 0xFFFF as f32; } }),


                        _ => panic!()
                    }
                },

                PixelFormat::RGB32f => {
                    /// Returns the current source pixel as `f32` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const f32, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFF as f32/3.0) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFFFF as f32/3.0) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3.0; }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = ((src!()[0] + src!()[1] + src!()[2]) / 3.0) as f64; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!()[2-i] * 0xFF as f32) as u8; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = (src!()[2-i] * 0xFF as f32) as u8; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!({ for i in 0..3 { dest!(3, u8)[i] = (src!()[i] * 0xFF as f32) as u8; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] * 0xFFFF as f32) as u16; } }),

                        PixelFormat::RGB64f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f64)[i] = src!()[i] as f64; } }),

                        _ => panic!()
                    }
                },

                PixelFormat::RGB64f => {
                    /// Returns the current source pixel as `f64` RGB values.
                    macro_rules! src { () => { unsafe { slice::from_raw_parts(self.pixels[src_ofs..].as_ptr() as *const f64, 3) } }};

                    match dest_img.pix_fmt {
                        PixelFormat::Mono8 => convert_whole_line!({ dest!(1, u8)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFF as f64/3.0) as u8; }),

                        PixelFormat::Mono16 => convert_whole_line!({ dest!(1, u16)[0] = ((src!()[0] + src!()[1] + src!()[2]) * 0xFFFF as f64/3.0) as u16; }),

                        PixelFormat::Mono32f => convert_whole_line!({ dest!(1, f32)[0] = ((src!()[0] + src!()[1] + src!()[2]) / 3.0) as f32; }),

                        PixelFormat::Mono64f => convert_whole_line!({ dest!(1, f64)[0] = (src!()[0] + src!()[1] + src!()[2]) / 3.0; }),

                        PixelFormat::BGRA8 =>
                            convert_whole_line!({
                                let bgra = dest!(4, u8);
                                bgra[3] = 0xFF;
                                for i in 0..3 { bgra[i] = (src!()[2-i] * 0xFF as f64) as u8; }
                            }),

                        PixelFormat::BGR8 =>
                            convert_whole_line!({
                                let bgr = dest!(3, u8);
                                for i in 0..3 { bgr[i] = (src!()[2-i] * 0xFF as f64) as u8; }
                            }),

                        PixelFormat::RGB8 =>
                            convert_whole_line!({ for i in 0..3 { dest!(3, u8)[i] = (src!()[i] * 0xFF as f64) as u8; } }),

                        PixelFormat::RGB16 =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, u16)[i] = (src!()[i] * 0xFFFF as f64) as u16; } }),

                        PixelFormat::RGB32f =>
                            convert_whole_line!(
                            { for i in 0..3 { dest!(3, f32)[i] = src!()[i] as f32; } }),

                        _ => panic!()
                    }
                },

                _ => panic!()
            }
        }
    }


    /// Returns a fragment of the image converted to the specified pixel format.
    ///
    /// The fragment to convert starts at `src_pos` in `&self` and has `width`x`height` pixels.
    /// Cropping is performed if necessary. If `&self` is in raw color format, the CFA pattern
    /// will be appropriately adjusted, depending on `src_pos`.
    ///
    pub fn convert_pix_fmt_of_subimage(&self,
                                       dest_pix_fmt: PixelFormat,
                                       src_pos: Point,
                                       width: u32,
                                       height: u32,
                                       demosaic_method: Option<DemosaicMethod>) -> Image {
        let mut new_pal: Option<Palette> = None;
        if self.pix_fmt == PixelFormat::Pal8 {
            new_pal = Some(self.palette.iter().next().unwrap().clone());
        }

        let mut dest_img = Image::new(width, height, None, dest_pix_fmt, new_pal, false);

        self.convert_pix_fmt_of_subimage_into(&mut dest_img, src_pos, Point{ x: 0, y: 0 }, width, height, demosaic_method);

        dest_img
    }


    /// Returns the image converted to the specified pixel format.
    #[must_use]
    pub fn convert_pix_fmt(&self,
                           dest_pix_fmt: PixelFormat,
                           demosaic_method: Option<DemosaicMethod>) -> Image {
        self.convert_pix_fmt_of_subimage(dest_pix_fmt, Point{ x: 0, y: 0 }, self.width, self.height, demosaic_method)
    }

    /// Returns a copy of image's fragment. The fragment boundaries may extend outside of the image.
    ///
    /// The fragment to copy is `width`x`height` pixels and starts at `src_pos`.
    /// If `clear_to_zero` is true, fragment's areas outside of the image will be cleared to zero.
    ///
    pub fn fragment_copy(
        &self,
        src_pos: &Point,
        width: u32,
        height: u32,
        clear_to_zero: bool
    ) -> Image {
        let mut dest_img = Image::new(width, height, None, self.pix_fmt, self.palette, clear_to_zero);
        self.resize_and_translate_into(&mut dest_img, *src_pos, width, height, Point{ x: 0, y: 0 }, clear_to_zero);
        dest_img
    }

    /// Copies (with cropping or padding) a fragment of image to another. There is no scaling.
    ///
    /// Pixel formats of source and destination must be the same.
    /// The fragment to copy is `width`x`height` pixels and starts at `src_pos` in `&self`
    /// and at `dest_pos` at `dest_img`. If `clear_to_zero` is true, `dest_img`'s areas not copied on
    /// will be cleared to zero.
    /// NOTE: care must be taken if pixel format is raw color (CFA). The caller may need to adjust
    /// the CFA pattern if source and destination X, Y offets are not simultaneously odd/even.
    ///
    pub fn resize_and_translate_into(
        &self,
        dest_img: &mut Image,
        src_pos: Point,
        width: u32,
        height: u32,
        dest_pos: Point,
        clear_to_zero: bool) {

        assert!(self.pix_fmt == dest_img.pix_fmt);

        let src_w = self.width;
        let src_h = self.height;
        let dest_w = dest_img.width;
        let dest_h = dest_img.height;

        let b_per_pix = bytes_per_pixel(self.pix_fmt);

        // Start and end (inclusive) coordinates to fill in the output image
        let mut dest_x_start = dest_pos.x;
        let mut dest_x_end = dest_pos.x + width as i32 - 1;

        let mut dest_y_start = dest_pos.y;
        let mut dest_y_end = dest_pos.y + height as i32 - 1;

        // Actual source coordinates to use
        let mut src_x_start = src_pos.x;
        let mut src_y_start = src_pos.y;

        // Perform any necessary cropping

        // Source image, left and top
        if src_pos.x < 0 {
            src_x_start -= src_pos.x;
            dest_x_start -= src_pos.x;
        }
        if src_pos.y < 0 {
            src_y_start -= src_pos.y;
            dest_y_start -= src_pos.y;
        }

        // Source image, right and bottom
        if src_pos.x + width as i32 > src_w as i32 {
            dest_x_end -= src_pos.x + width as i32 - src_w as i32;
        }

        if src_pos.y + height as i32 > src_h as i32 {
            dest_y_end -= src_pos.y + height as i32 - src_h as i32;
        }

        // Destination image, left and top
        if dest_x_start < 0 {
            src_x_start -= dest_x_start;
            dest_x_start = 0;
        }
        if dest_y_start < 0 {
            src_y_start -= dest_y_start;
            dest_y_start = 0;
        }

        // Destination image, right and bottom
        if dest_x_end >= dest_w as i32 {
            dest_x_end = dest_w as i32 - 1;
        }
        if dest_y_end >= dest_h as i32 {
            dest_y_end = dest_h as i32 - 1;
        }

        if dest_y_end < dest_y_start || dest_x_end < dest_x_start {
            // Nothing to copy

            if clear_to_zero {
                // Also works for floating-point pixels; all zero bits = 0.0
                unsafe { ptr::write_bytes(dest_img.pixels[..].as_mut_ptr(), 0, (dest_img.width * dest_img.height) as usize); }
            }
            return;
        }

        if clear_to_zero {
            // Unchanged lines at the top
            for y in 0..dest_y_start as u32 {
                unsafe { ptr::write_bytes(dest_img.line_raw_mut(y).as_mut_ptr(), 0, dest_img.bytes_per_line); }
            }

            // Unchanged lines at the bottom
            for y in dest_y_end as u32 + 1 .. dest_img.height {
                unsafe { ptr::write_bytes(dest_img.line_raw_mut(y).as_mut_ptr(), 0, dest_img.bytes_per_line); }
            }

            for y in dest_y_start as u32 .. dest_y_end as u32 + 1 {
                // Columns to the left of the target area
                unsafe { ptr::write_bytes(dest_img.line_raw_mut(y).as_mut_ptr(), 0, dest_x_start as usize * b_per_pix); }

                // Columns to the right of the target area
                let dest_ptr: *mut u8 = dest_img.pixels[y as usize * dest_img.bytes_per_line + (dest_x_end as usize + 1) * b_per_pix ..].as_mut_ptr();
                unsafe { ptr::write_bytes(dest_ptr, 0, (dest_img.width as usize - 1 - dest_x_end as usize) * b_per_pix); }
            }
        }

        // Copy the pixels line by line
        for y in dest_y_start .. dest_y_end  + 1 {
            let line_copy_bytes = (dest_x_end - dest_x_start + 1) as usize * b_per_pix;

            let src_line_ofs = src_x_start as usize * b_per_pix;
            let src_line: &[u8] = &self.line_raw((y - dest_y_start + src_y_start) as u32)[range!(src_line_ofs, line_copy_bytes)];

            let dest_line_ofs = dest_x_start as usize * b_per_pix;
            let dest_line: &mut [u8] = &mut dest_img.line_raw_mut(y as u32)[range!(dest_line_ofs, line_copy_bytes)];

            dest_line.copy_from_slice(src_line);
        }
    }

    /// Changes endianess of multi-byte pixel values (does nothing for 8-bits-per-channel formats).
    pub fn reverse_byte_order(&mut self) {
        match self.pix_fmt {
            PixelFormat::Mono16
            | PixelFormat::RGB16
            | PixelFormat::RGBA16
            | PixelFormat::CfaRGGB16
            | PixelFormat::CfaGRBG16
            | PixelFormat::CfaGBRG16
            | PixelFormat::CfaBGGR16 => change_endianess(self.pixels_mut::<u16>()),

            PixelFormat::Mono32f
            | PixelFormat::RGB32f => change_endianess(self.pixels_mut::<f32>()),

            PixelFormat::Mono64f
            | PixelFormat::RGB64f => change_endianess(self.pixels_mut::<f64>()),

            _ => ()
        }
    }

    /// Returns the image scaled to new size using linear interpolation.
    pub fn scale(&self, new_width: u32, new_height: u32) -> Image {
        assert!(new_width > 0 && new_height > 0);
        assert!(self.pix_fmt != PixelFormat::Pal8);

        let mut new_img = Image::new(new_width, new_height, None, self.pix_fmt, None, false);

        match self.pix_fmt {
            PixelFormat::Mono8 |
            PixelFormat::RGB8 |
            PixelFormat::BGR8 |
            PixelFormat::BGRA8 => scale_image::<u8>(&self, &mut new_img),

            PixelFormat::Mono16 |
            PixelFormat::RGB16 |
            PixelFormat::RGBA16 => scale_image::<u16>(&self, &mut new_img),

            PixelFormat::Mono32f |
            PixelFormat::RGB32f => scale_image::<f32>(&self, &mut new_img),

            PixelFormat::Mono64f |
            PixelFormat::RGB64f => scale_image::<f64>(&self, &mut new_img),

            _ => panic!("Scaling not supported for pixel format {:?}.", self.pix_fmt)
        }

        new_img
    }
}

trait FromF64 {
    fn from_f64(f: f64) -> Self;
}

impl FromF64 for u8 {
    fn from_f64(f: f64) -> Self { f as u8 }
}

impl FromF64 for u16 {
    fn from_f64(f: f64) -> Self { f as u16 }
}

impl FromF64 for f32 {
    fn from_f64(f: f64) -> Self { f as f32 }
}

impl FromF64 for f64 {
    fn from_f64(f: f64) -> Self { f }
}

fn scale_image<T: Any + Clone + Default + Into<f64> + FromF64>(src: &Image, dest: &mut Image) {
    assert!(src.pix_fmt != PixelFormat::Pal8);
    assert!(src.pix_fmt == dest.pix_fmt);

    let num_channels = num_channels(src.pix_fmt);

    let dest_width = dest.width;
    let dest_height = dest.height;

    for y in 0..dest_height {
        let src_y = y as f64 * src.height as f64 / dest_height as f64;
        let src_y_lo = src_y.floor() as u32;
        let src_row_lo = src.line::<T>(src_y_lo);
        let src_row_hi = src.line::<T>((src_y_lo + 1).min(src.height - 1));
        let ty = src_y.fract();

        let dest_row = dest.line_mut::<T>(y);

        for x in 0..dest_width {
            let src_x = x as f64 * src.width as f64 / dest_width as f64;
            let src_x_lo = src_x.floor() as usize;
            let src_x_hi = (src_x_lo + 1).min(src.width as usize - 1);
            let tx = src_x.fract();

            for ch in 0..num_channels {
                let v_00 = Into::<f64>::into(src_row_lo[src_x_lo + ch].clone());
                let v_10 = Into::<f64>::into(src_row_lo[src_x_hi + ch].clone());
                let v_11 = Into::<f64>::into(src_row_hi[src_x_hi + ch].clone());
                let v_01 = Into::<f64>::into(src_row_hi[src_x_lo + ch].clone());

                dest_row[x as usize + ch] = FromF64::from_f64(
                    v_00 * (1.0 - tx) * (1.0 - ty) +
                    v_10 * tx * (1.0 - ty) +
                    v_11 * tx * ty +
                    v_01 * (1.0 - tx) * ty);
            }
        }
    }
}

fn change_endianess<T>(words: &mut [T]) {
    let len = std::mem::size_of::<T>();

    for word in words.iter_mut() {
        let bytes_1 = unsafe { std::slice::from_raw_parts_mut(word as *mut T as *mut u8, len) };
        let bytes_2 = unsafe { std::slice::from_raw_parts_mut(word as *mut T as *mut u8, len) };
        for i in 0..len/2 {
            std::mem::swap(
                unsafe { bytes_1.get_unchecked_mut(len - i - 1) },
                unsafe { bytes_2.get_unchecked_mut(i) }
            );
        }
    }
}

impl Clone for Image {
    fn clone(&self) -> Image {
        let new_pixels = self.pixels.clone();
        Image::new_from_pixels(self.width, self.height, Some(self.bytes_per_line), self.pix_fmt, self.palette, new_pixels)
    }
}
