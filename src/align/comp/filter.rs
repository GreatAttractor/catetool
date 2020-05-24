//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Image filters.
//!

use crate::image::{Image, PixelFormat};
use cgmath::Vector2;

#[must_use]
pub fn gaussian_blur(image: &Image, sigma: f32) -> Image {
    let mut result = image.clone();
    gaussian_blur_in_place(&mut result, sigma);
    result
}

pub fn gaussian_blur_in_place(image: &mut Image, sigma: f32) {
    assert!(image.pixel_format() == PixelFormat::Mono32f);
    assert!(sigma >= 0.5);

    let width = image.width() as usize;
    let height = image.height() as usize;

    let yvv = calc_yvv_coefficients(sigma);

    // convolve rows
    for y in 0..height {
        let line = image.line_mut::<f32>(y as u32);
        // perform forward filtering
        yvv_filter_values(line, width, 1, 1, &yvv);
        // perform backward filtering
        yvv_filter_values(line, width, -1, 1, &yvv);
    }

    // convolve columns
    let vals_per_line = image.values_per_line::<f32>();
    let pixels = image.pixels_mut::<f32>();
    for x in 0..width {
        // perform forward filtering
        yvv_filter_values(&mut pixels[x..], height, 1, vals_per_line, &yvv);
        // perform backward filtering
        yvv_filter_values(&mut pixels[x..], height, -1, vals_per_line, &yvv);
    }
}

/// Young & van Vliet recursive Gaussian coefficients.
struct YvVCoefficients {
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    B: f32
}

#[must_use]
fn calc_yvv_coefficients(sigma: f32) -> YvVCoefficients {
    let q = if sigma >= 0.5 && sigma <= 2.5 {
        3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).sqrt()
    } else {
        0.98711 * sigma - 0.9633
    };

    let b0 = 1.57825 + 2.44413 * q + 1.4281 * q.powi(2) + 0.422205 * q.powi(3);
    let b1 = 2.44413 * q + 2.85619 * q.powi(2) + 1.26661 * q.powi(3);
    let b2 = -1.4281 * q.powi(2) - 1.26661 * q.powi(3);
    let b3 = 0.422205 * q.powi(3);
    let B = 1.0 - ((b1 + b2 + b3) / b0);

    YvVCoefficients{ b0, b1, b2, b3, B }
}

/// Performs a Young & van Vliet approximated recursive Gaussian filtering of values in one direction.
///
/// # Parameters
///
/// * `direction` - 1: filter forward, -1: filter backward; if -1, processing starts at the last element.
///
fn yvv_filter_values(
    values: &mut [f32],
    count: usize,
    direction: i32,
    stride: usize,
    yvv: &YvVCoefficients
) {
    assert!(direction == 1 || direction == -1);

    let b0_inv = 1.0 / yvv.b0;

    // start index to process (inclusive)
    let start_idx = if direction == 1 {
        0
    } else {
        stride * (count - 1)
    };

    // assume that border values extend beyond the array
    let mut prev1 = values[start_idx];
    let mut prev2 = values[start_idx];
    let mut prev3 = values[start_idx];

    let mut i = start_idx as i32;
    for _ in 0..count {
        let next = yvv.B * values[i as usize] + (yvv.b1 * prev1 + yvv.b2 * prev2 + yvv.b3 * prev3) * b0_inv;

        prev3 = prev2;
        prev2 = prev1;
        prev1 = next;

        values[i as usize] = next;

        i += direction * stride as i32;
    }
}
