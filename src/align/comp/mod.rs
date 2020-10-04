//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Computations module.
//!

use cgmath::{InnerSpace, Vector2, Basis2, Rad, Rotation, Rotation2};
use crate::image::{Image, PixelFormat, get_matching_mono_format};
use crate::image::point::{Point, Rect};
use std::any::Any;
use std::mem::MaybeUninit;
use num_complex::Complex32;
use rayon::prelude::*;

mod fft;
pub mod filter;

/// Margin (in pixels) expected for bilinear interpolation.
pub const MARGIN: i32 = 3;

pub struct Disk {
    pub center: Vector2<i32>,
    pub radius: u32
}

/// Iterates from `min` (inclusive) to `max` (inclusive).
struct FloatRangeIter {
    min: f32,
    max: f32,
    step: f32,
    value: f32,
    step_count: isize
}

impl FloatRangeIter {
    fn new(min: f32, max: f32, step: f32) -> FloatRangeIter {
        FloatRangeIter{ min, max, step, value: min, step_count: -1 }
    }
}

impl Iterator for FloatRangeIter {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        self.step_count += 1;
        self.value = self.min + self.step * self.step_count as f32;
        if self.value <= self.max {
            Some(self.value)
        } else {
            None
        }
    }
}

pub trait PixelMask {
    fn is_inside_rect(&self, rect: &Rect, mask_pos: &Vector2<i32>) -> bool;
    fn points(&self) -> &[Vector2<i32>];
    fn xmin(&self) -> i32;
    fn xmax(&self) -> i32;
    fn ymin(&self) -> i32;
    fn ymax(&self) -> i32;
}

pub trait FromF32 {
    fn from_f32(f: f32) -> Self;
}

impl FromF32 for u8 {
    fn from_f32(f: f32) -> Self { f as u8 }
}

impl FromF32 for u16 {
    fn from_f32(f: f32) -> Self { f as u16 }
}

impl FromF32 for f32 {
    fn from_f32(f: f32) -> Self { f }
}

impl FromF32 for f64 {
    fn from_f32(f: f32) -> Self { f as f64 }
}

fn cross(v1: &Vector2<i32>, v2: &Vector2<i32>) -> i32 {
    v1.x * v2.y - v1.y * v2.x
}

pub fn to_f32(v: &Vector2<i32>) -> Vector2<f32> { Vector2{ x: v.x as f32, y: v.y as f32 } }

pub fn to_i32(v: &Vector2<f32>) -> Vector2<i32> { Vector2{ x: v.x as i32, y: v.y as i32 } }

pub fn to_point(v: &Vector2<i32>) -> Point { Point{ x: v.x, y: v.y } }

fn magnitude2(v: &Vector2<i32>) -> i32 { v.x.pow(2) + v.y.pow(2) }

/// Returns `image` downscaled so that the larger dimension becomes `max_desired_dimension`.
///
/// Returns (downscaled image, scaling factor).
///
pub fn downscale(image: &Image, max_desired_dimension: u32) -> (Image, f64) {
    let w = image.width();
    let h = image.height();
    if w > h {
        let factor = max_desired_dimension as f64 / w as f64;
        (image.scale(max_desired_dimension, (h as f64 * factor) as u32), factor)
    } else {
        let factor = max_desired_dimension as f64 / h as f64;
        (image.scale((w as f64 * factor) as u32, max_desired_dimension), factor)
    }
}

/// Returns accumulator of circle Hough transform of `image`.
pub fn hough_circle_accumulator(image: &Image, threshold: f32, radius: u32) -> Image {
    assert!(image.pixel_format() == PixelFormat::Mono32f);

    let circle_points = rasterize_circle(radius);
    let src_pixels = image.pixels::<f32>();
    let img_values_per_line = image.values_per_line::<f32>();

    let mut accumulator = Image::new(image.width(), image.height(), None, PixelFormat::Mono32f, None, true);
    let accum_values_per_line = accumulator.values_per_line::<f32>();
    let accum_pixels = accumulator.pixels_mut::<f32>();

    for y in 0..image.height() {
        let src_offs = y as usize * img_values_per_line;
        for x in 0..image.width() {
            if src_pixels[src_offs + x as usize] >= threshold {
                for c_point in circle_points.iter()
                    .map(|p| Vector2{ x: p.x + x as i32, y: p.y + y as i32 })
                    .filter(|p| p.x >= 0 && p.x < image.width() as i32 &&
                                p.y >= 0 && p.y < image.height() as i32) {

                    accum_pixels[c_point.y as usize * accum_values_per_line + c_point.x as usize] += 1.0;
                }
            }
        }
    }

    accumulator
}

// Returns circle points clockwise (in a right-handed coordinate system), starting from the leftmost point.
fn rasterize_circle(radius: u32) -> Vec<Vector2<i32>> {
    let mut octant = vec![];

    let mut point = Vector2{ x: -(radius as i32), y: 0 };

    // Is `Some` if the point having x=y belongs to the circle.
    let mut diagonal_point: Option<Vector2<i32>> = None;

    while -point.x > point.y {
        point.x += 1;
        point.y += 1;
        if point.x.pow(2) + point.y.pow(2) < radius.pow(2) as i32 {
            point.x -= 1;
        }
        if point.x.abs() == point.y.abs() {
            diagonal_point = Some(point);
        } else {
            octant.push(point);
        }
    }

    let mut points = vec![];

    // Order of filling octants:
    //
    //               y
    //               ^
    //               |
    //         oct2  |  oct3
    //        +      ^       +
    //     oct_1     |     oct4
    // ----+---------0------------+----->x
    //     oct8      |     oct5
    //        +      |       +
    //         oct7  |  oct6
    //               |


    points.push(Vector2{ x: -(radius as i32), y: 0 });
    points.extend_from_slice(&octant); // octant 1
    match diagonal_point { Some(ref p) => points.push(*p), _ => () }
    points.extend(octant.iter().rev().map(|p| Vector2{ x: -p.y, y: -p.x })); // octant 2
    points.push(Vector2{ x: 0, y: radius as i32 });
    points.extend(octant.iter().map(|p| Vector2{ x: p.y, y: -p.x })); // octant 3
    match diagonal_point { Some(ref p) => points.push(Vector2{ x: -p.x, y: p.y }), _ => () }
    points.extend(octant.iter().rev().map(|p| Vector2{ x: -p.x, y: p.y })); // octant 4
    points.push(Vector2{ x: radius as i32, y: 0 });
    points.extend(octant.iter().map(|p| Vector2{ x: -p.x, y: -p.y })); // octant 5
    match diagonal_point { Some(ref p) => points.push(Vector2{ x: -p.x, y: -p.y }), _ => () }
    points.extend(octant.iter().rev().map(|p| Vector2{ x: p.y, y: p.x })); // octant 6
    points.push(Vector2{ x: 0, y: -(radius as i32) });
    points.extend(octant.iter().map(|p| Vector2{ x: -p.y, y: p.x })); // octant 7
    match diagonal_point { Some(ref p) => points.push(Vector2{ x: p.x, y: -p.y }), _ => () }
    points.extend(octant.iter().rev().map(|p| Vector2{ x: p.x, y: -p.y })); // octant 8

    points
}

/// (Exact) Blackman window function; for `x` = 0.0 returns 0.0, for `x` = 1.0 returns 1.0.
///
/// To be used on relevant image fragments before feeding the images to `determine_translation`.
///
pub fn blackman_window(x: f32) -> f32 {
    let a0 = 7938.0 / 18608.0;
    let a1 = 9240.0 / 18608.0;
    let a2 = 1430.0 / 18608.0;

    a0 - a1 * (std::f32::consts::PI * x).cos() + a2 * (2.0 * std::f32::consts::PI * x).cos()
}

pub fn calc_image_fft(image: &Image) -> Vec<Complex32> {
    assert!(image.pixel_format() == PixelFormat::Mono32f);
    assert!(fft::is_power_of_2(image.width() as usize) &&
            fft::is_power_of_2(image.height() as usize));

    let mut output = vec![MaybeUninit::<Complex32>::uninit(); (image.width() * image.height()) as usize];
    fft::fft_2d_uninit(
        image.height() as usize,
        image.width() as usize,
        image.values_per_line::<f32>(),
        image.pixels::<f32>(),
        &mut output
    );
    let output = unsafe { std::mem::transmute::<_, Vec<Complex32>>(output) };

    output
}

/// Determines translation vector (with sub-pixel accuracy) between specified images using phase correlation.
///
/// If the images are non-periodic, the fragments relevant for comparison should be multiplied by a window function
/// (e.g. `blackman_window`). The images have to have the same dimensions; their width and height must be powers of 2.
///
pub fn determine_translation(img_1: &Image, img_2: &Image) -> Vector2<f32> {
    assert!(img_1.width() == img_2.width());
    assert!(img_1.height() == img_2.height());
    assert!(fft::is_power_of_2(img_1.width() as usize) &&
            fft::is_power_of_2(img_1.height() as usize));

    let width = img_1.width() as usize;
    let height = img_1.height() as usize;

    let mut fft_1 = vec![MaybeUninit::<Complex32>::uninit(); width * height];
    fft::fft_2d_uninit(
        height,
        width,
        img_1.values_per_line::<f32>(),
        img_1.pixels::<f32>(),
        &mut fft_1
    );
    let fft_1 = unsafe { std::mem::transmute::<_, Vec<Complex32>>(fft_1) };

    let mut fft_2 = vec![MaybeUninit::<Complex32>::uninit(); width * height];
    fft::fft_2d_uninit(
        height,
        width,
        img_2.values_per_line::<f32>(),
        img_2.pixels::<f32>(),
        &mut fft_2
    );
    let fft_2 = unsafe { std::mem::transmute::<_, Vec<Complex32>>(fft_2) };

    determine_translation_from_image_fft(width, height, &fft_1, &fft_2)
}

/// Determines (using phase correlation) the vector by which image 2 (given by its
/// discrete Fourier transform 'fft_2') is translated w.r.t. image 1 ('fft_1').
///
/// Subpixel translation detection based on:
///
///   Extension of Phase Correlation to Subpixel Registration
///   Hassan Foroosh, Josiane B. Zerubia, Marc Berthod
///
pub fn determine_translation_from_image_fft(
    width: usize,
    height: usize,
    fft_1: &[Complex32],
    fft_2: &[Complex32]
) -> Vector2<f32> {
    assert!(fft_1.len() == width * height);
    assert!(fft_2.len() == width * height);

    // cross-power spectrum
    let mut cps = vec![MaybeUninit::<Complex32>::uninit(); width * height];
    fft::calc_cross_power_spectrum_2d_uninit(fft_1, fft_2, &mut cps);
    let cps = unsafe { std::mem::transmute::<_, Vec<Complex32>>(cps) };

    // cross-correlation
    let mut cc = vec![MaybeUninit::<Complex32>::uninit(); width * height];
    fft::fft_2d_inverse_uninit(height, width, &cps, &mut cc);
    let cc = unsafe { std::mem::transmute::<_, Vec<Complex32>>(cc) };

    // find the max real value in `cc` (and its position)
    let (cc_peak, max_x, max_y) = cc.iter().enumerate()
        .max_by(|a, b| a.1.re.partial_cmp(&b.1.re).unwrap())
        .map(|(i, value)| (value.re, (i % width) as i32, (i / width) as i32)
    ).unwrap();

    let translation_x: i32 = if max_x < width as i32 / 2 { max_x } else { max_x - width as i32 };
    let translation_y: i32 = if max_y < height as i32 / 2 { max_y } else { max_y - height as i32 };

    // find the subpixel translation
    let mut tx_frac: f32 = 0.0;
    let mut ty_frac: f32 = 0.0;

    macro_rules! clamp_w { ($k:expr) => { ($k + width as i32) % width as i32 } }
    macro_rules! clamp_h { ($k:expr) => { ($k + height as i32) % height as i32 } }

    let cc_x_hi = cc[(clamp_w!(max_x + 1) + max_y * width as i32) as usize].re;
    let cc_x_lo = cc[(clamp_w!(max_x - 1) + max_y * width as i32) as usize].re;

    let cc_y_hi = cc[(max_x + clamp_h!(max_y + 1) * width as i32) as usize].re;
    let cc_y_lo = cc[(max_x + clamp_h!(max_y - 1) * width as i32) as usize].re;

    if cc_x_hi > cc_x_lo {
        let dx1 = cc_x_hi / (cc_x_hi + cc_peak);
        let dx2 = cc_x_hi / (cc_x_hi - cc_peak);

        if dx1 > 0.0 && dx1 < 1.0 {
            tx_frac = dx1;
        } else if dx2 > 0.0 && dx2 < 1.0 {
            tx_frac = dx2;
        }
    }
    else {
        let dx1 = cc_x_lo / (cc_x_lo + cc_peak);
        let dx2 = cc_x_lo / (cc_x_lo - cc_peak);

        if dx1 > 0.0 && dx1 < 1.0 {
            tx_frac = -dx1;
        } else if dx2 > 0.0 && dx2 < 1.0 {
            tx_frac = -dx2;
        }
    }

    if cc_y_hi > cc_y_lo {
        let dy1 = cc_y_hi / (cc_y_hi + cc_peak);
        let dy2 = cc_y_hi / (cc_y_hi - cc_peak);

        if dy1 > 0.0 && dy1 < 1.0 {
            ty_frac = dy1;
        } else if dy2 > 0.0 && dy2 < 1.0 {
            ty_frac = dy2;
        }
    } else {
        let dy1 = cc_y_lo / (cc_y_lo + cc_peak);
        let dy2 = cc_y_lo / (cc_y_lo - cc_peak);

        if dy1 > 0.0  && dy1 < 1.0 {
            ty_frac = -dy1;
        } else if dy2 > 0.0 && dy2 < 1.0 {
            ty_frac = -dy2;
        }
    }

    Vector2{
        x: translation_x as f32 + tx_frac,
        y: translation_y as f32 + ty_frac
    }
}

/// Returns a copy of `image` with the pixels specified by `ring` (relative to `center`)
/// scaled around `center` by `scale`.
#[must_use]
pub fn scale_ring<T: Any + Copy + Default + Into<f32> + FromF32>(
    image: &Image,
    center: &Vector2<f32>,
    scale: f32,
    ring: &RingMask
) -> Image {
    assert!(image.pixel_format() == get_matching_mono_format::<T>());

    assert!(ring.is_inside_rect(&image.img_rect().inflate(-MARGIN), &to_i32(center)));

    let src_pixels = image.pixels::<T>();
    let src_values_per_line = image.values_per_line::<T>();

    let mut output = Image::new(image.width(), image.height(), None, get_matching_mono_format::<T>(), None, true);
    let dest_values_per_line = output.values_per_line::<T>();
    let dest_pixels = output.pixels_mut::<T>();

    for p in ring.points().iter() {
        // perform bilinear interpolation in `source` at the location corresponding to `x`, `y`
        let Vector2{ x: src_x, y: src_y } = to_f32(&p) / scale + center;

        let src_x_lo = src_x.floor() as usize;
        let src_y_lo = src_y.floor() as usize;

        let tx = src_x.fract() as f32;
        let ty = src_y.fract() as f32;

        let v_00 = src_pixels[src_x_lo +     src_y_lo * src_values_per_line];
        let v_10 = src_pixels[src_x_lo + 1 + src_y_lo * src_values_per_line];

        let v_11 = src_pixels[src_x_lo + 1 + (src_y_lo + 1) * src_values_per_line];
        let v_01 = src_pixels[src_x_lo +     (src_y_lo + 1) * src_values_per_line];

        dest_pixels[(p.x + center.x as i32) as usize +
                    (p.y + center.y as i32) as usize * dest_values_per_line
        ] = FromF32::from_f32(
            Into::<f32>::into(v_00) * (1.0 - tx) * (1.0 - ty) +
            Into::<f32>::into(v_10) * tx         * (1.0 - ty) +
            Into::<f32>::into(v_11) * tx         * ty +
            Into::<f32>::into(v_01) * (1.0 - tx) * ty
        );
    }

    output
}

/// Returns a copy of `image` with the pixels specified by `ring` (relative to `center`) rotated clockwise by `angle`
/// and scaled around `center` by `scale`.
pub fn rotate_ring<T: Any + Copy + Default + Into<f32> + FromF32>(
    image: &Image,
    angle: f32,
    center: &Vector2<f32>,
    scale: f32,
    ring: &RingMask
) -> Image {
    assert!(image.pixel_format() == get_matching_mono_format::<T>());

    assert!(ring.is_inside_rect(&image.img_rect().inflate(-MARGIN), &to_i32(center)));

    let rot: Basis2<f32> = Rotation2::from_angle(Rad(-angle as f32));

    let src_pixels = image.pixels::<T>();
    let src_values_per_line = image.values_per_line::<T>();

    let mut output = Image::new(image.width(), image.height(), None, get_matching_mono_format::<T>(), None, true);
    let dest_values_per_line = output.values_per_line::<T>();
    let dest_pixels = output.pixels_mut::<T>();

    // Note: using unchecked accesses below has barely any effect on the performance.

    for p in ring.points().iter() {
        // perform bilinear interpolation in `source` at the location corresponding to `x`, `y`
        let Vector2{ x: src_x, y: src_y } = rot.rotate_vector(to_f32(&p)) / scale + center;

        let src_x_lo = src_x.floor() as usize;
        let src_y_lo = src_y.floor() as usize;

        let tx = src_x.fract() as f32;
        let ty = src_y.fract() as f32;

        let v_00 = src_pixels[src_x_lo +     src_y_lo * src_values_per_line];
        let v_10 = src_pixels[src_x_lo + 1 + src_y_lo * src_values_per_line];

        let v_11 = src_pixels[src_x_lo + 1 + (src_y_lo + 1) * src_values_per_line];
        let v_01 = src_pixels[src_x_lo +     (src_y_lo + 1) * src_values_per_line];

        dest_pixels[(p.x + center.x as i32) as usize +
                    (p.y + center.y as i32) as usize * dest_values_per_line
        ] = FromF32::from_f32(
            Into::<f32>::into(v_00) * (1.0 - tx) * (1.0 - ty) +
            Into::<f32>::into(v_10) * tx         * (1.0 - ty) +
            Into::<f32>::into(v_11) * tx         * ty +
            Into::<f32>::into(v_01) * (1.0 - tx) * ty
        );
    }

    output
}

/// List of points belonging to a ring centered at (0, 0).
pub struct RingMask {
    r_inner: i32,
    r_outer: i32,
    points: Vec<Vector2<i32>>
}

impl PixelMask for RingMask {
    fn is_inside_rect(&self, rect: &Rect, mask_pos: &Vector2<i32>) -> bool {
        mask_pos.x - self.r_outer >= rect.x &&
        mask_pos.x + self.r_outer < rect.x + rect.width as i32 &&
        mask_pos.y - self.r_outer >= rect.y &&
        mask_pos.y + self.r_outer < rect.y as i32 + rect.height as i32
    }

    fn points(&self) -> &[Vector2<i32>] { &self.points }

    fn xmin(&self) -> i32 { -self.r_outer }

    fn xmax(&self) -> i32 { self.r_outer }

    fn ymin(&self) -> i32 { -self.r_outer }

    fn ymax(&self) -> i32 { self.r_outer }
}

impl RingMask {
    pub fn new(r_inner: i32, r_outer: i32) -> RingMask {
        assert!(r_inner >= 0);
        assert!(r_outer > 0);
        assert!(r_outer > r_inner);

        let mut points = vec![];
        for y in -r_outer..=r_outer {
            for x in -r_outer..r_outer {
                let dist_sq = x.pow(2) + y.pow(2);
                if dist_sq >= r_inner.pow(2) && dist_sq <= r_outer.pow(2) {
                    points.push(Vector2{ x, y });
                }
            }
        }

        RingMask{ r_inner, r_outer, points }
    }

    pub fn r_inner(&self) -> i32 { self.r_inner }

    pub fn r_outer(&self) -> i32 { self.r_outer }

    pub fn points(&self) -> &[Vector2<i32>] { &self.points }
}

/// List of points belonging to a rectangle with disk cut out at (0, 0).
pub struct NoDiskMask {
    radius: i32,
    width: i32,
    height: i32,
    points: Vec<Vector2<i32>>
}

impl PixelMask for NoDiskMask {
    fn is_inside_rect(&self, rect: &Rect, mask_pos: &Vector2<i32>) -> bool {
        panic!("Not implemented yet.");
    }

    fn points(&self) -> &[Vector2<i32>] { &self.points }

    fn xmin(&self) -> i32 { -self.width / 2 }

    fn xmax(&self) -> i32 { -self.width / 2 + self.width - 1 }

    fn ymin(&self) -> i32 { -self.height / 2 }

    fn ymax(&self) -> i32 { -self.height / 2 + self.height - 1 }
}

impl NoDiskMask {
    pub fn new(radius: i32, width: i32, height: i32) -> NoDiskMask {
        assert!(radius >= 0);
        assert!(width > 0);
        assert!(height > 0);
        let mut points = vec![];
        for y in -height / 2..(-height / 2 + height) {
            for x in -width / 2..(-width / 2 + width) {
                if x.pow(2) + y.pow(2) >= radius.pow(2) {
                    points.push(Vector2{ x, y });
                }
            }
        }

        NoDiskMask{ radius, width, height, points }
    }

    pub fn points(&self) -> &[Vector2<i32>] { &self.points }

    pub fn radius(&self) -> i32 { self.radius }

    pub fn width(&self) -> i32 { self.width }

    pub fn height(&self) -> i32 { self.height }
}

/// Calculates sum of absolute pixel differences between images inside the specified ring.
fn calc_sum_of_abs_diffs<M: PixelMask>(img1: &Image, img2: &Image, center1: &Vector2<i32>, center2: &Vector2<i32>, mask: &M) -> u64 {
    assert!(mask.is_inside_rect(&img1.img_rect(), &center1));
    assert!(mask.is_inside_rect(&img2.img_rect(), &center2));

    assert!(img1.pixel_format() == PixelFormat::Mono8 && img2.pixel_format() == PixelFormat::Mono8);

    let pixels1 = img1.pixels::<u8>();
    let values_per_line_1 = img1.bytes_per_line();

    let pixels2 = img2.pixels::<u8>();
    let values_per_line_2 = img2.bytes_per_line();

    let sum_diff: u64 = mask.points().iter().fold(0, |sum, p| {
        sum + unsafe { (
            *pixels1.get_unchecked((p.x + center1.x) as usize + (p.y + center1.y) as usize * values_per_line_1) as i16 -
            *pixels2.get_unchecked((p.x + center2.x) as usize + (p.y + center2.y) as usize * values_per_line_2) as i16
        ).abs() as u64 }
    });

    sum_diff
}

/// Returns sum of absolute differences of ring pixel values in the images, assuming that in `img2` the ring interior
/// is scaled by `scale`. Also returns the scaled image (only pixels within the ring).
///
/// # Parameters
///
/// * `img1` - Image (`Mono8`) to compare to.
/// * `img2` - Image (`Mono8`) to be compared.
/// * `scale` - Scale factor of the ring interior in `img2`.
/// * `center1` - Ring position in `img1`.
/// * `center2` - Ring position in `img2`.
///
/// Returns: (sum of abs. differences of pixel values, scaled ring of `img2`)
///
fn check_scale(
    img1: &Image,
    img2: &Image,
    center1: &Vector2<i32>,
    center2: &Vector2<f32>,
    scale: f32,
    ring: &RingMask
) -> (f64, Image) {
    assert!(ring.is_inside_rect(&img2.img_rect().inflate(-MARGIN), &to_i32(center2)));

    let scaled = scale_ring::<u8>(&img2, &center2, scale, ring);
    let sum_diff = calc_sum_of_abs_diffs_subpixel::<u8, _>(img1, &scaled, ring, center1, center2, 0.0);

    (sum_diff, scaled)
}

/// Finds the relative scale of `Mono8` images; multithreaded.
pub fn find_relative_scale(
    img1: &Image,
    img2: &Image,
    center1: &Vector2<i32>,
    center2: &Vector2<f32>,
    min_scale: f32,
    max_scale: f32,
    scale_step: f32,
    ring: &RingMask
) -> f32 {
    assert!(max_scale > min_scale && scale_step > 0.0);

    let num_test_scales = ((max_scale - min_scale) / scale_step) as usize;
    // each element: (sum of pixel differences, scale)
    let mut results: Vec<Option<(f64, f32)>> = vec![None; num_test_scales];

    results.par_iter_mut().enumerate().for_each(|(i, result)| {
        let scale = min_scale + i as f32 * scale_step;
        let (sum_diffs, scaled_ring) = check_scale(img1, img2, center1, center2, scale, ring);
        *result = Some((sum_diffs, scale));
    });

    let best_result: (f64, f32) = results.iter_mut().min_by(
        |r1, r2| r1.as_ref().unwrap().0.partial_cmp(&r2.as_ref().unwrap().0).unwrap()
    ).unwrap().take().unwrap();

    1.0 / best_result.1
}

/// Returns sum of absolute differences of ring pixel values in the images, assuming that in `img2` the ring is rotated
/// clockwise by `angle`. Also returns the rotated image (only pixels within the ring).
///
/// # Parameters
///
/// * `img1` - Image (`Mono8`) to compare to.
/// * `img2` - Image (`Mono8`) to be compared.
/// * `angle` - Angle in radians by which to rotate the ring in `img2`.
/// * `center1` - Ring position in `img1`.
/// * `center2` - Ring position in `img2`.
/// * `scale2` - Scale factor of `img2` relative to `img1`.
///
/// Returns: (sum of abs. differences of pixel values, rotated ring of `img2`)
///
fn check_rotation_angle(
    img1: &Image,
    img2: &Image,
    angle: f32,
    center1: &Vector2<i32>,
    center2: &Vector2<f32>,
    scale2: f32,
    ring: &RingMask
) -> (f64, Image) {
    assert!(ring.is_inside_rect(&img2.img_rect().inflate(-MARGIN), &to_i32(center2)));

    let rotated = rotate_ring::<u8>(&img2, angle, &center2, scale2, ring);

    //let sum_diff = calc_sum_of_abs_diffs(img1, &rotated, center1, &to_i32(&center2), ring);
    let sum_diff = calc_sum_of_abs_diffs_subpixel::<u8, _>(img1, &rotated, ring, center1, center2, 0.0);

    (sum_diff, rotated)
}

/// Finds the relative angle between `Mono8` images; multithreaded.
///
/// Returns (angle, rotated ring of `img2`).
///
pub fn find_relative_angle(
    img1: &Image,
    img2: &Image,
    center1: &Vector2<i32>,
    center2: &Vector2<f32>,
    min_angle: f32,
    max_angle: f32,
    angle_step: f32,
    scale: f32,
    ring: &RingMask
) -> (f32, Image) {
    assert!(max_angle > min_angle && angle_step > 0.0);

    let num_test_angles = ((max_angle - min_angle) / angle_step) as usize;
    // each element: (sum of pixel differences, angle, rotated ring)
    let mut results: Vec<Option<(f64, f32, Image)>> = vec![None; num_test_angles];

    results.par_iter_mut().enumerate().for_each(|(i, result)| {
        let angle = min_angle + i as f32 * angle_step;
        let (sum_diffs, rotated_ring) = check_rotation_angle(img1, img2, angle, center1, center2, scale, ring);
        *result = Some((sum_diffs, angle, rotated_ring));
    });

    let best_result: (f64, f32, Image) = results.iter_mut().min_by(
        |r1, r2| r1.as_ref().unwrap().0.partial_cmp(&r2.as_ref().unwrap().0).unwrap()
    ).unwrap().take().unwrap();

    (best_result.1, best_result.2)
}

/// Finds the translation vector between images with 1-pixel accuracy. Only pixels in `mask` are compared.
pub fn find_translation_vector<T: PixelMask>(
    img1: &Image,
    img2: &Image,
    center1: &Vector2<i32>,
    center2: &Vector2<i32>,
    mask: &T,
    search_radius: i32,
    initial_step: i32
) -> Vector2<i32> {
    assert!(search_radius > 0 && initial_step > 0);

    assert!(center1.x + mask.xmin() - search_radius >= 0 &&
            center1.x + mask.xmax() + search_radius < img1.width() as i32);
    assert!(center1.y + mask.ymin() - search_radius >= 0 &&
            center1.y + mask.ymax() + search_radius < img1.height() as i32);

    assert!(mask.is_inside_rect(&img2.img_rect(), &center2));

    struct SearchRange {
        // inclusive
        pub xmin: i32,
        pub ymin: i32,
        // exclusive
        pub xmax: i32,
        pub ymax: i32
    };

    // start the search with a coarse step, then continue around the best position using a repeatedly smaller step,
    // until it becomes 1
    let mut search_step = initial_step;

    let mut search_range = SearchRange{
        xmin: center1.x - search_radius,
        ymin: center1.y - search_radius,
        xmax: center1.x + search_radius,
        ymax: center1.y + search_radius
    };

    let mut best_pos = Vector2{ x: 0, y: 0 };

    while search_step > 0 {
        let mut min_sq_diff_sum = u64::max_value();

        //TODO: do it in parallel
        // (x, y) = position of `ring` in `img1` for which a block match test is performed
        let mut y = search_range.ymin;
        while y < search_range.ymax {
            let mut x = search_range.xmin;
            while x < search_range.xmax {
                let sum_abs_diffs = calc_sum_of_abs_diffs(img1, img2, &Vector2{ x, y }, center2, mask);

                if sum_abs_diffs < min_sq_diff_sum {
                    min_sq_diff_sum = sum_abs_diffs;
                    best_pos = Vector2{ x, y };
                }

                x += search_step;
            }
            y += search_step;
        }

        search_range.xmin = best_pos.x - search_step;
        search_range.ymin = best_pos.y - search_step;
        search_range.xmax = best_pos.x + search_step;
        search_range.ymax = best_pos.y + search_step;

        search_step /= 2;
    }

    *center1 - best_pos
}

/// Returns a `Mono32f` image with pixel values changing radially around `center` according to `window_func`.
///
/// If `mask_radius` is 0, the values are 1.0 at `center` and 0.0 at the borders of a `width`×`height` rectangle.
///
/// If `mask_radius` is non-zero, for each line segment from `center` to the rectangle's border, the values are 0.0
/// at `mask_radius` from `center`, 1.0 at midway to the border, and 0.0 at the border.
///
/// `window_func` returns for 0.0 for argument 0.0, and 1.0 for argument 1.0.
///
pub fn create_radial_window(
    width: u32,
    height: u32,
    center: &Vector2<i32>,
    mask_radius: u32,
    window_func: fn(f32) -> f32
) -> Image {
    assert!(Rect{ x: 0, y: 0, width, height }.contains_point(&Point::new(center.x, center.y)));

    let mut image = Image::new(width, height, None, PixelFormat::Mono32f, None, true);

    // Vectors from `center` towards the image corners; used to simplify the search for image border intersection.
    //
    // (the drawing uses right-handed coord. system, does not impact the results)
    //
    //  (0,h)          (w,h)
    //   +--------------+
    //   |              |
    //   |   *center    |
    //   |              |
    //   |       *p     |
    //   +--------------+
    //  (0,0)         (0,w)
    //
    //  Vector (p-center) has its cross-product with `to_00` negative, and with `to_0w`: positive,
    // so the intersection of (p-center) with image border is at the bottom edge.
    //
    //  The other edges are tested for intersection similarly, each using two subsequent `to_##` vectors.
    //
    let to_00 = Vector2{ x: 0 - center.x,            y: 0 - center.y };
    let to_w0 = Vector2{ x: width as i32 - center.x, y: 0 - center.y };
    let to_wh = Vector2{ x: width as i32 - center.x, y: height as i32 - center.y };
    let to_0h = Vector2{ x: 0 - center.x,            y: height as i32 - center.y };

    //TODO: parallellize
    for y in 0..height as i32 {
        let img_line = image.line_mut::<f32>(y as u32);
        for x in 0..width as i32 {
            let pvec = Vector2{ x, y } - *center;

            if magnitude2(&pvec) <= mask_radius.pow(2) as i32 {
                img_line[x as usize] = 0.0;
                continue;
            }

            // find the intersection point between `pvec` and image border
            let intercept = if pvec.x == 0 && pvec.y == 0 {
                Vector2{ x: center.x, y: 0 }
            } else if cross(&pvec, &to_00) <= 0 && cross(&pvec, &to_w0) >= 0 {
                *center + (pvec * (-center.y)) / (y - center.y)
            } else if cross(&pvec, &to_w0) <= 0 && cross(&pvec, &to_wh) >= 0 {
                *center + (pvec * (width as i32 - center.x)) / (x - center.x)
            } else if cross(&pvec, &to_wh) <= 0 && cross(&pvec, &to_0h) >= 0 {
                *center + (pvec * (height as i32 - center.y)) / (y - center.y)
            } else {
                *center + (pvec * (- center.x)) / (x - center.x)
            };

            // maps (x, y) to the square extending between [-1; 1], with `center` at (0, 0)
            let square_11_map = |p: Vector2<i32>| {
                let mapped_x = if p.x == center.x {
                    0.0
                } else if p.x > center.x {
                    (p.x - center.x) as f32 / (width as i32 - center.x) as f32
                } else {
                    (p.x - center.x) as f32 / center.x as f32
                };

                let mapped_y = if p.y == center.y {
                    0.0
                } else if p.y > center.y {
                    (p.y - center.y) as f32 / (height as i32 - center.y) as f32
                } else {
                    (p.y - center.y) as f32 / center.y as f32
                };

                Vector2{ x: mapped_x, y: mapped_y }
            };

            // smooth map from [-1,1] square to unit circle
            // (http://mathproofs.blogspot.com/2005/07/mapping-square-to-circle.html)
            let curvilinear_map = |p: Vector2<f32>| {
                Vector2{
                    x: p.x * (1.0 - p.y.powi(2) / 2.0).sqrt(),
                    y: p.y * (1.0 - p.x.powi(2) / 2.0).sqrt()
                }
            };

            // point at mask's circumference between `center` and (x, y)
            let q1 = square_11_map( *center + to_i32(&(mask_radius as f32 * to_f32(&pvec).normalize())));
            // current image point
            let q2 = square_11_map(Vector2{ x, y });
            // point at image border
            let q3 = square_11_map(intercept);

            // relative radial position between the mask's circumference (0.0) and the image border (1.0),
            // after curvilinear mapping
            let rel_radial_pos: f32 = if magnitude2(&pvec) <= mask_radius.pow(2) as i32 {
                0.0
            } else {
                let s1 = curvilinear_map(q1);
                let s2 = curvilinear_map(q2);
                let s3 = curvilinear_map(q3);

                ((s2 - s1).magnitude2() / (s3 - s1).magnitude2()).sqrt()
            };

            if mask_radius == 0 {
                img_line[x as usize] = window_func(1.0 - rel_radial_pos);
            } else {
                img_line[x as usize] = window_func(2.0 * (-(rel_radial_pos - 0.5).abs() + 0.5));
            }
        }
    }

    image
}

/// Returns `image` padded in width and height to the nearest powers of 2.
pub fn pad_for_phase_correlation(image: &Image) -> Image {
    let new_width = if fft::is_power_of_2(image.width() as usize) {
        image.width()
    } else {
        1 << (fft::quick_log2(image.width() as usize) + 1)
    };
    let new_height = if fft::is_power_of_2(image.height() as usize) {
        image.height()
    } else {
        1 << (fft::quick_log2(image.height() as usize) + 1)
    };

    let mut result = Image::new(new_width, new_height, None, image.pixel_format(), None, true);
    image.resize_and_translate_into(&mut result, Point::zero(), image.width(), image.height(), Point::zero(), true);

    result
}

/// Creates a window having 0.0 at the inner and outer boundaries of `ring`, and 1.0 in the middle.
/// Values at intermediate positions follow `window_func`.
pub fn create_ring_window(ring: &RingMask, window_func: fn(f32) -> f32) -> Image {
    let r_out = ring.r_outer();
    let dim = (ring.r_outer() * 2 + 1) as u32;
    let mut window = Image::new(dim, dim, None, PixelFormat::Mono32f, None, true);
    let values_per_line = window.values_per_line::<f32>();
    let pixels = window.pixels_mut::<f32>();

    for p in ring.points() {
        let r = ((magnitude2(&p) as f32).sqrt() - ring.r_inner() as f32) / (r_out - ring.r_inner()) as f32;

        pixels[(p.x + r_out) as usize + (p.y + r_out) as usize * values_per_line] =
            window_func(2.0 * (-(r - 0.5).abs() + 0.5));
    }

    window
}

pub fn multiply_by_window(image: &mut Image, window: &Image) {
    assert!(image.pixel_format() == PixelFormat::Mono32f && window.pixel_format() == PixelFormat::Mono32f);

    let w = image.width();
    let h = image.height();
    for y in 0..h {
        let line = image.line_mut::<f32>(y);
        let wnd_line = window.line::<f32>(y);
        for x in 0..w {
            line[x as usize] *= wnd_line[x as usize];
        }
    }
}

pub fn unsharp_mask(image: &mut Image, sigma: f32, amount: f32) {
    let blurred = filter::gaussian_blur(image, sigma);

    for y in 0..image.height() {
        let line = image.line_mut::<f32>(y);
        let blurred_line = blurred.line::<f32>(y);
        for (val, blurred) in line.iter_mut().zip(blurred_line.iter()) {
            *val = amount * (*val) + (1.0 - amount) * blurred;
        }
    }
}

/// Normalizes pixel values assuming there is exponential falloff away from image center.
pub fn radial_exp_brighten(image: &mut Image) {
    assert!(image.pixel_format() == PixelFormat::Mono32f);
    let w = image.width();
    let h = image.height();
    for y in 0..h {
        let line = image.line_mut::<f32>(y);
        for x in 0..w {
            let dist_from_center = (((x as i32 - w as i32 / 2).pow(2) + (y as i32 - h as i32 / 2).pow(2)) as f32).sqrt();
            line[x as usize] *= (dist_from_center * 4.0e-3).exp();
        }
    }
}

fn calc_sum_of_abs_diffs_subpixel<T, M>(
    image1: &Image,
    image2: &Image,
    mask: &M,
    pos1: &Vector2<i32>,
    pos2: &Vector2<f32>,
    threshold: f32
) -> f64
where T: Any + Copy + Default + Into<f32>, M: PixelMask + Sync {

    assert!(mask.is_inside_rect(&image1.img_rect(), pos1));
    assert!(mask.is_inside_rect(&image2.img_rect().inflate(-MARGIN), &to_i32(pos2)));

    let pixels1 = image1.pixels::<T>();
    let values_per_line_1 = image1.values_per_line::<T>();

    let pixels2 = image2.pixels::<T>();
    let values_per_line_2 = image2.values_per_line::<T>();

    // Note: using unchecked accesses below has barely any effect on the performance.

    let sum_diff: f64 = mask.points().iter().fold(0.0, |sum, p| {
        let value1 = Into::<f32>::into(pixels1[(p.x + pos1.x) as usize + (p.y + pos1.y) as usize * values_per_line_1]);
        if value1 < threshold { return sum; }

        let x2 = p.x as f32 + pos2.x;
        let y2 = p.y as f32 + pos2.y;

        let x2_lo = x2.floor() as usize;
        let y2_lo = y2.floor() as usize;

        let tx = x2.fract();
        let ty = y2.fract();

        // perform bilinear interpolation in `image2` at the location corresponding to `x2`, `y2`
        let v_00 = pixels2[x2_lo +     y2_lo * values_per_line_2];
        let v_10 = pixels2[x2_lo + 1 + y2_lo * values_per_line_2];
        let v_11 = pixels2[x2_lo + 1 + (y2_lo + 1) * values_per_line_2];
        let v_01 = pixels2[x2_lo +     (y2_lo + 1) * values_per_line_2];

        let value2 =
            Into::<f32>::into(v_00) * (1.0 - tx) * (1.0 - ty) +
            Into::<f32>::into(v_10) * tx         * (1.0 - ty) +
            Into::<f32>::into(v_11) * tx         * ty +
            Into::<f32>::into(v_01) * (1.0 - tx) * ty;

        sum + (value1 - value2).abs() as f64
    });

    sum_diff
}

pub fn add_image(
    dest: &mut Image,
    dest_add_count: &mut Vec<usize>,
    src: &Image,
    replace_nan: Option<f32>,
    src_excl_disk: Option<Disk>
) {
    assert!(dest.pixel_format() == PixelFormat::Mono32f && src.pixel_format() == PixelFormat::Mono32f);
    assert!(dest.img_rect() == src.img_rect());
    assert!(dest_add_count.len() == (src.width() * src.height()) as usize);

    for y in 0..src.height() {
        let dest_line = dest.line_mut::<f32>(y);
        let src_line = src.line::<f32>(y);
        for x in 0..src.width() {
            if let Some(ref disk) = src_excl_disk {
                if (x as i32 - disk.center.x).pow(2) + (y as i32 - disk.center.y).pow(2) <= disk.radius.pow(2) as i32 {
                    continue;
                }
            }

            dest_line[x as usize] += match replace_nan{
                Some(rep) => { let sval = src_line[x as usize]; if sval.is_nan() { rep } else { sval } },
                None => src_line[x as usize]
            };

            dest_add_count[(x + y * src.width()) as usize] += 1;
        }
    }
}

/// Finds the translation vector between images with sub-pixel accuracy. Only pixels in `mask` are compared.
///
/// Multithreaded.
///
pub fn find_translation_vector_subpixel2<T, M>(
    image1: &Image,
    image2: &Image,
    pos1: &Vector2<i32>,
    pos2: &Vector2<f32>,
    mask: &M,
    search_radius: f32,
    initial_step: f32,
    min_step: f32,
    threshold: f32
) -> Vector2<f32>
where T: Any + Copy + Default + Into<f32>, M: PixelMask + Sync {

    assert!(search_radius > 0.0 && initial_step > 0.0 && min_step > 0.0);

    struct SearchRange {
        pub xmin: f32,
        pub ymin: f32,
        pub xmax: f32,
        pub ymax: f32
    };

    // start the search with a coarse step, then continue around the best position
    // using a repeatedly smaller step, until it becomes `min_step`
    let mut search_step = initial_step;

    let mut search_range = SearchRange{
        xmin: pos2.x - search_radius,
        ymin: pos2.y - search_radius,
        xmax: pos2.x + search_radius,
        ymax: pos2.y + search_radius
    };

    let mut best_pos = Vector2{ x: 0.0, y: 0.0 };

    while search_step >= min_step {
        let search_positions: Vec<Vector2<f32>> =
            FloatRangeIter::new(search_range.xmin, search_range.xmax, search_step)
            .map(|x| FloatRangeIter::new(search_range.ymin, search_range.ymax, search_step)
                .map(move |y| Vector2{ x, y })
            ).flatten().collect();

        // element [i] corresponds to `search_positions[i]`
        let mut sum_abs_diffs_at_search_pos = vec![0.0; search_positions.len()];

        sum_abs_diffs_at_search_pos.par_iter_mut().enumerate().for_each(|(i, sum_abs_diffs)| {
            *sum_abs_diffs = calc_sum_of_abs_diffs_subpixel::<T, _>(
                image1, image2, mask, pos1, &search_positions[i], threshold
            );
        });

        let best_idx = sum_abs_diffs_at_search_pos
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;

        best_pos = search_positions[best_idx];

        search_range.xmin = best_pos.x - search_step;
        search_range.ymin = best_pos.y - search_step;
        search_range.xmax = best_pos.x + search_step;
        search_range.ymax = best_pos.y + search_step;

        if search_step > min_step && search_step / 2.0 < min_step {
            search_step = min_step;
        } else {
            search_step /= 2.0;
        }
    }

    best_pos - pos2
}

#[must_use]
pub fn scale_and_clip_pixel_values(mut image: Image, scale: f32, max: f32) -> Image {
    assert!(image.pixel_format() == PixelFormat::Mono32f);
    for p in image.pixels_mut::<f32>() {
        *p *= scale;
        if *p > max { *p = max; }
    }

    image
}

/// Brings all pixel values to [0.0, 1.0], scales by `scale` > 0, and clips to [0.0, 1.0].
#[must_use]
pub fn normalize_pixel_values(image: Image, scale: f32) -> Image {
    assert!(image.pixel_format() == PixelFormat::Mono32f);

    let min = *image.pixels::<f32>().iter().min_by(|a, b|
        if a.is_nan() { std::cmp::Ordering::Greater } else if b.is_nan() { std::cmp::Ordering::Less } else {
            a.partial_cmp(&b).unwrap()
        }
    ).unwrap();

    let max = *image.pixels::<f32>().iter().max_by(|a, b|
        if a.is_nan() { std::cmp::Ordering::Less } else if b.is_nan() { std::cmp::Ordering::Greater } else {
            a.partial_cmp(&b).unwrap()
        }
    ).unwrap();

    let mut result = image.clone();
    for p in result.pixels_mut::<f32>() {
        *p = (*p - min) / (max - min) * scale;
        if *p > 1.0 { *p = 1.0; }
    }

    result
}

/// Finds lunar disk's center.
pub fn find_lunar_disk_center(image: &Image, radius: f32, background_threshold: f32) -> Vector2<i32> {
    assert!(image.pixel_format() == PixelFormat::Mono32f);

    let ray_start = Vector2{ x: image.width() as i32 / 2, y: image.height() as i32 / 2 };

    let mut limb_points: Vec<Vector2<f32>> = vec![];

    for dx in -1..=1 {
        for dy in -1..=1 {
            if dx == 0 && dy == 0 { continue; }

            let ray_values = get_ray_values(image, &ray_start, dx, dy);
            if let Some(limb_point_idx) = find_lunar_limb_point(ray_values, background_threshold) {
                limb_points.push(Vector2{
                    x: ray_start.x as f32 + (limb_point_idx as i32 * dx) as f32,
                    y: ray_start.y as f32 + (limb_point_idx as i32 * dy) as f32
                });
            }
        }
    }

    let center = fit_circle_to_points(&limb_points, radius);
    Vector2{ x: center.x as i32, y: center.y as i32 }
}

fn find_lunar_limb_point(ray: Vec<f32>, background_threshold: f32) -> Option<usize> {
    const TRANSITION_WIDTH: usize = 6/*10*/;
    const SOLAR_DISK_BRIGHTNESS: f32 = 20_000.0;

    if ray.iter().find(|&&x| x > SOLAR_DISK_BRIGHTNESS).is_some() {
        // do not use ray that pass through the unobscured solar disk
        return None;
    }

    // We declare the disk->background transition to start when subsequent `TRANSITION_WIDTH` ray points
    // all have brightness above `background_threshold`.

    let mut brightest = 0.0f32;
    let mut last_within = TRANSITION_WIDTH - 1;
    let mut last_above = 0usize;
    for i in TRANSITION_WIDTH..ray.len() - 1 {
        if ray[i] <= background_threshold {
            last_within = i;
        } else {
            last_above = i;
        }

        if last_above > last_within && (last_above - last_within) >= TRANSITION_WIDTH {
            return Some(last_within + 1);
        }

        brightest = brightest.max(ray[i]);
    }

    let mut error_msg = format!(
        "Lunar limb point not found; did not find {} consecutive pixels above background threshold {} in the ray:\n",
        TRANSITION_WIDTH, background_threshold
    );
    let ray_max_idx = ray.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap();

    error_msg += "[..., ";
    for i in 0.max(ray_max_idx as i32 - TRANSITION_WIDTH as i32)..(ray_max_idx + TRANSITION_WIDTH).min(ray.len()) as i32 {
        error_msg += &format!("{}, ", ray[i as usize]);
    }
    error_msg += "...]\nConsider specifying a lower background threshold.";

    panic!("{}", error_msg);
}


/// Gets pixel values of a ray from `start` to image border, having direction -1⩽dx⩽1, -1⩽dy⩽1.
fn get_ray_values(image: &Image, start: &Vector2<i32>, dx: i32, dy: i32) -> Vec<f32> {
    assert!(image.pixel_format() == PixelFormat::Mono32f);
    assert!(dx.abs() <= 1 && dy.abs() <= 1);
    assert!(dx != 0 || dy.abs() != 0);

    let mut values = vec![];

    let mut x = start.x;
    let mut y = start.y;
    while x >=0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
        values.push(image.line::<f32>(y as u32)[x as usize]);

        x += dx;
        y += dy;
    }

    values
}

#[allow(non_snake_case)]
#[must_use]
fn fit_circle_to_points(points: &Vec<Vector2<f32>>, radius: f32) -> Vector2<f32> {
    // current approximation of the center
    let mut center = Vector2{
        x: points.iter().fold(0.0, |acc, p| acc + p.x) / points.len() as f32,
        y: points.iter().fold(0.0, |acc, p| acc + p.y) / points.len() as f32
    };

    // residuals; element [i] corresponds to points[i]
    let mut R = vec![0.0; points.len()];

    // Jacobian; element [i] contains the partial derivatives of the i-th residual
    // with respect to `center.x` and `center.y`
    let mut J = vec![[0.0; 2]; points.len()];

    const NUM_GAUSS_NEWTON_ITERATIONS: usize = 8;
    for _ in 0..NUM_GAUSS_NEWTON_ITERATIONS {
        for (i, point) in points.iter().enumerate() {
            // Functions (residuals) r_i (which we seek to minimize) are the differences between the radius
            // and the i-th point distance from the circle center:
            //
            //   r_i(cx, cy, r) = sqrt((x_i - cx)^2 + (y_i - cy)^2) - r
            //
            // The Jacobian matrix entries are defined as:
            //
            //   J(i, j) = partial derivative of r_i(cx, cy, r) with respect to j-th parameter
            //   (where parameters cx, cy, r are numbered 0, 1, 2)
            //
            // which gives:
            //
            //   J(i, 0) = (cx - x_i) / sqrt((x_i - cx)^2 + (y_i - cy)^2)
            //   J(i, 1) = (cy - y_i) / sqrt((x_i - cx)^2 + (y_i - cy)^2)

            let dist = ((center.x - point.x).powi(2) + (center.y - point.y).powi(2)).sqrt();
            R[i] = dist - radius;
            J[i][0] = (center.x - point.x) / dist;
            J[i][1] = (center.y - point.y) / dist;
        }

        // Jᵀ ⨉ J   (2⨉2 matrix)
        let Jt_J: [[f32; 2]; 2] = {
            let mut result_00 = 0.0;
            let mut result_11 = 0.0;
            let mut result_10_01 = 0.0;
            for j_row in &J {
                result_00 += j_row[0].powi(2);
                result_11 += j_row[1].powi(2);
                result_10_01 += j_row[0] * j_row[1];
            }

            [[result_00, result_10_01], [result_10_01, result_11]]
        };

        // Jᵀ ⨉ R   (2⨉1 matrix)
        let Jt_R: [f32; 2] = {
            let mut result_00 = 0.0;
            let mut result_01 = 0.0;
            for (j_row, residual) in J.iter().zip(R.iter()) {
                result_00 += j_row[0] * residual;
                result_01 += j_row[1] * residual;
            }

            [result_00, result_01]
        };

        // (Jᵀ ⨉ J)⁻¹   (2⨉2 matrix)
        let Jt_J_inv: [[f32; 2]; 2] = {
            let det = Jt_J[0][0] * Jt_J[1][1] - Jt_J[0][1] * Jt_J[1][0];
            [
                [ Jt_J[1][1] / det, -Jt_J[0][1] / det],
                [-Jt_J[1][0] / det,  Jt_J[0][0] / det]
            ]
        };

        // (Jᵀ ⨉ J)⁻¹ ⨉ (Jᵀ ⨉ R)   (2⨉1 matrix)
        let Jt_J_inv_Jt_R: [f32; 2] = [
            Jt_J_inv[0][0] * Jt_R[0] + Jt_J_inv[0][1] * Jt_R[1],
            Jt_J_inv[1][0] * Jt_R[0] + Jt_J_inv[1][1] * Jt_R[1],
        ];

        center.x -= Jt_J_inv_Jt_R[0];
        center.y -= Jt_J_inv_Jt_R[1];
    }

    center
}

#[must_use]
pub fn translate_pixels<T>(image: &Image, translation: &Vector2<f32>) -> Image
where T: Any + Copy + Default + Into<f32> + FromF32 {

    assert!(image.pixel_format() == get_matching_mono_format::<T>());

    let src_pixels = image.pixels::<T>();
    let src_values_per_line = image.values_per_line::<T>();

    let mut output = Image::new(image.width(), image.height(), None, get_matching_mono_format::<T>(), None, true);

    let y_min = (translation.y as i32 + MARGIN)
        .max(MARGIN);
    let y_max = (translation.y as i32 + image.height() as i32 - MARGIN)
        .min(image.height() as i32 - MARGIN);

    let x_min = (translation.x as i32 + MARGIN)
        .max(MARGIN);
    let x_max = (translation.x as i32 + image.width() as i32 - MARGIN)
        .min(image.width() as i32 - MARGIN);

    for dest_y in y_min..y_max {
        let dest_line = output.line_mut::<T>(dest_y as u32);
        for (dest_x, dest_val) in dest_line.iter_mut().skip(x_min as usize).take((x_max - x_min) as usize).enumerate() {

            // perform bilinear interpolation in `image` at the location corresponding to `x`, `y`
            let src_x = dest_x as f32 - translation.x;
            let src_y = dest_y as f32 - translation.y;

            let src_x_lo = src_x.floor() as usize;
            let src_y_lo = src_y.floor() as usize;

            let tx = src_x.fract() as f32;
            let ty = src_y.fract() as f32;

            let v_00 = src_pixels[src_x_lo +     src_y_lo * src_values_per_line];
            let v_10 = src_pixels[src_x_lo + 1 + src_y_lo * src_values_per_line];

            let v_11 = src_pixels[src_x_lo + 1 + (src_y_lo + 1) * src_values_per_line];
            let v_01 = src_pixels[src_x_lo +     (src_y_lo + 1) * src_values_per_line];

            *dest_val = FromF32::from_f32(
                Into::<f32>::into(v_00) * (1.0 - tx) * (1.0 - ty) +
                Into::<f32>::into(v_10) * tx         * (1.0 - ty) +
                Into::<f32>::into(v_11) * tx         * ty +
                Into::<f32>::into(v_01) * (1.0 - tx) * ty
            );
        }
    }

    output
}
