//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Fast Fourier Transform functions.
//!

use num_complex::Complex32;
use num_traits::identities::{One, Zero};
use rayon::prelude::*;
use std::mem::MaybeUninit;

/// Returns floor(log2(n)).
pub fn quick_log2(mut n: usize) -> usize
{
    if n == 0 {
        return 0;
    }

    let mut result = 0;
    while n > 0 {
        n >>= 1;
        result += 1;
    }

    result - 1
}

/// Calculates twiddle factors for FFT input of `fft_size`.
pub fn calc_twiddle_factors(fft_size: usize, inverse: bool) -> Vec<Complex32> {
    let mut result = vec![Complex32::zero(); quick_log2(fft_size) as usize + 1];

    let mut denominator = fft_size;
    for n in (0..=quick_log2(fft_size)).rev() {
        result[n as usize] = if inverse {
            (2.0 * std::f32::consts::PI * Complex32::i() / denominator as f32).exp()
        } else {
            (-2.0 * std::f32::consts::PI * Complex32::i() / denominator as f32).exp()
        };
        denominator >>= 1;
    }

    result
}

/// Wrapper for `fft1_1d_uninit`.
pub fn fft_1d<T: Copy + Into<Complex32>>(
    n: usize,
    input: &[T],
    output: &mut [Complex32],
    i_step: usize,
    o_step: usize,
    twiddles: &[Complex32]
) {
    fft_1d_uninit(
        n,
        input,
        unsafe { std::mem::transmute::<_, &mut [MaybeUninit<Complex32>]>(output) },
        i_step,
        o_step,
        twiddles
    );
}

/// Calculates 1-dimensional discrete Fourier transform or its inverse (not normalized by `input`'s length,
/// the caller must do this).
///
/// After the calculations complete, all values in `output` are initialized and can be `transmute`d to `Complex32`.
///
/// # Parameters
///
/// * `n` - Number of values to calculate.
/// * `input` - Input values; same length as `output` (must be a power of 2).
/// * `output` - Output values; same length as `input` (must be a power of 2).
/// * `i_step` - Input step; external callers must specify 1.
/// * `o_step` - Output step; external callers must specify 1.
/// * `twiddle_factors` - Last element is the twiddle factor corresponding to input's length `n`, i.e. exp(-2*π*i / n)
///     (or exp(2*π*i / n) for inverse transform). Second-to-last element must be the next lower twiddle factor,
///     i.e. exp(±2*π*i / (n/2).
///
/// # Postconditions
///
/// All values in `output` are initialized.
///
pub fn fft_1d_uninit<T: Copy + Into<Complex32>>(
    n: usize,
    input: &[T],
    output: &mut [MaybeUninit<Complex32>],
    i_step: usize,
    o_step: usize,
    twiddles: &[Complex32]
) {
    //TODO: use unchecked accesses (and benchmark)
    if n == 1 {
        output[0] = MaybeUninit::new(input[0].into());
    } else {
        fft_1d_uninit(
            n / 2,
            input,
            output,
            2 * i_step,
            o_step,
            &twiddles[..twiddles.len() - 1]
        );

        fft_1d_uninit(
            n / 2,
            &input[i_step..],
            &mut output[n / 2 * o_step..],
            2 * i_step,
            o_step,
            &twiddles[..twiddles.len() - 1]
        );

        // initial twiddle factor
        let t_factor_0 = twiddles.last().unwrap();

        let mut t_factor = Complex32::one();

        for k in 0..n / 2 {
            let t = unsafe { output[k * o_step].assume_init() };
            let h = t_factor * unsafe { output[(k + n / 2) * o_step].assume_init() };

            output[k * o_step] = MaybeUninit::new(t + h);
            output[(k + n / 2) * o_step] = MaybeUninit::new(t - h);

            t_factor *= t_factor_0; // in effect, t_factor = exp(-2*π*i * k/n)
        }
    }
}

pub fn is_power_of_2(n: usize) -> bool {
    n.count_ones() == 1
}

/// Wrapper for `fft_2d_uninit`.
pub fn fft_2d(
    rows: usize,
    cols: usize,
    input_stride: usize,
    input: &[f32],
    output: &mut [Complex32]
) {
    fft_2d_uninit(
        rows,
        cols,
        input_stride,
        input,
        unsafe { std::mem::transmute::<_, &mut [MaybeUninit<Complex32>]>(output) }
    );
}

/// Return multiple mutable references to `slice` for concurrent mutable access;
/// DO NOT try to access the same elements from different references.
unsafe fn to_multiple_refs_mut<'a, T>(slice: &'a mut [T], count: usize) -> Vec<&'a mut [T]> {
    let mut result = vec![];
    for _ in 0..count {
        result.push(std::slice::from_raw_parts_mut(slice.as_mut_ptr(), slice.len()));
    }

    result
}

/// Calculates 2-dimensional discrete Fourier transform; multithreaded.
///
/// Uses the row-column algorithm.
///
/// # Parameters
///
/// * `rows` - Number of rows (must be a power of 2).
/// * `cols` - Number of columns (must be a power of 2).
/// * `input_stride` - Actual length (number of `f32`s) of an input row (including padding, if any).
/// * `input` - Input values (`rows` * `input_stride` elements).
/// * `output` - Output values (`rows` * `cols` elements).
///
/// # Postconditions
///
/// All values in `output` are initialized.
///
pub fn fft_2d_uninit(
    rows: usize,
    cols: usize,
    input_stride: usize,
    input: &[f32],
    output: &mut [MaybeUninit<Complex32>]
) {
    assert!(is_power_of_2(rows) && is_power_of_2(cols));
    assert!(input.len() == rows * input_stride);
    assert!(output.len() == rows * cols);
    assert!(rows > 0 && cols > 0);

    let max_dim = rows.max(cols);
    let twiddles = calc_twiddle_factors(max_dim, false);

    // calculate 1-dimensional transforms of all the rows
    let mut fft_rows: Vec<MaybeUninit<Complex32>> = vec![MaybeUninit::uninit(); rows * cols];

    // it is OK to have multiple mutable refs.; each parallel call to `fft_1d_uninit` modifies different elements
    let mut fft_rows_refs = unsafe { to_multiple_refs_mut(&mut fft_rows, rows) };
    fft_rows_refs.par_iter_mut().enumerate().for_each(|(k, fft_rows)| {
        fft_1d_uninit(
            cols,
            &input[k * input_stride..],
            &mut fft_rows[k * cols..],
            1,
            1,
            &twiddles[..quick_log2(cols) + 1]
        );
    });
    let fft_rows = unsafe { std::mem::transmute::<_, Vec<Complex32>>(fft_rows) };

    // calculate 1-dimensional transforms of all columns in `fft_rows` to get the final result
    let mut output_refs = unsafe { to_multiple_refs_mut(output, cols) };
    output_refs.par_iter_mut().enumerate().for_each(|(k, output)| {
        fft_1d_uninit(
            rows,
            &fft_rows[k..],
            &mut output[k..],
            cols,
            cols,
            &twiddles[..quick_log2(rows) + 1]
        );
    });
}

/// Wrapper for `fft_2d_inverse_uninit`.
pub fn fft_2d_inverse(
    rows: usize,
    cols: usize,
    input: &[Complex32],
    output: &mut [Complex32]
) {
    fft_2d_inverse_uninit(
        rows,
        cols,
        input,
        unsafe { std::mem::transmute::<_, &mut [MaybeUninit<Complex32>]>(output) }
    );
}

/// Calculates 2-dimensional inverse discrete Fourier transform; multithreaded.
///
/// Uses the row-column algorithm. Afterwards all values in `output` are initialized.
///
/// # Parameters
///
/// * `rows` - Number of rows (must be a power of 2).
/// * `cols` - Number of columns (must be a power of 2).
/// * `input` - Input values (`rows` * `cols` elements).
/// * `output` - Output values (`rows` * `cols` elements).
///
/// # Postconditions
///
/// All values in `output` are initialized.
///
pub fn fft_2d_inverse_uninit(
    rows: usize,
    cols: usize,
    input: &[Complex32],
    output: &mut [MaybeUninit<Complex32>]
) {
    assert!(is_power_of_2(rows) && is_power_of_2(cols));
    assert!(input.len() == rows * cols);
    assert!(output.len() == rows * cols);
    assert!(rows > 0 && cols > 0);

    let max_dim = rows.max(cols);
    let twiddles = calc_twiddle_factors(max_dim, true);

    // calculate 1-dimensional inverse transforms of all the rows
    let mut fft_rows: Vec<MaybeUninit<Complex32>> = vec![MaybeUninit::uninit(); rows * cols];

    // it is OK to have multiple mutable refs.; each parallel call to `fft_1d_uninit` modifies different elements
    let mut fft_rows_refs = unsafe { to_multiple_refs_mut(&mut fft_rows, rows) };
    fft_rows_refs.par_iter_mut().enumerate().for_each(|(k, fft_rows)| {
        fft_1d_uninit(
            cols,
            &input[k * cols..],
            &mut fft_rows[k * cols..],
            1,
            1,
            &twiddles[..quick_log2(cols) + 1]
        );
    });
    let mut fft_rows = unsafe { std::mem::transmute::<_, Vec<Complex32>>(fft_rows) };

    for c in &mut fft_rows {
        *c /= cols as f32;
    }

    // calculate 1-dimensional inverse transforms of all columns in `fft_rows` to get the final result
    let mut output_refs = unsafe { to_multiple_refs_mut(output, cols) };
    output_refs.par_iter_mut().enumerate().for_each(|(k, output)| {
        fft_1d_uninit(
            rows,
            &fft_rows[k..],
            &mut output[k..],
            cols,
            cols,
            &twiddles[..quick_log2(rows) + 1]
        );
    });

    for o_ref in output.iter_mut() {
        let c = &mut unsafe { o_ref.assume_init() };
        *c /= rows as f32;
    }
}

/// Calculates cross-power spectrum of two 2D discrete Fourier transforms.
///
/// # Postconditions
///
/// All values in `output` are initialized.
///
pub fn calc_cross_power_spectrum_2d_uninit(
    fft_1: &[Complex32],
    fft_2: &[Complex32],
    output: &mut [MaybeUninit<Complex32>]
) {
    assert!(fft_1.len() == fft_2.len());
    assert!(output.len() == fft_1.len());

    for (i, out) in output.iter_mut().enumerate() { //TODO: use `rayon`
        let mut cpow = fft_1[i].conj() * fft_2[i];
        let mag = cpow.norm();
        if mag > 1.0e-8 {
            cpow /= mag;
        }
        *out = MaybeUninit::new(cpow);
    }
}
