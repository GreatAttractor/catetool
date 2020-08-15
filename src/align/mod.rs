//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Image alignment module.
//!

use cgmath::{Basis2, Vector2, Rotation, Rotation2, Rad, Zero};
use crate::image;
use crate::image::{Image, PixelFormat, Point};
use crate::logging::Logger;

mod comp;

pub use comp::{to_point, to_i32, scale_and_clip_pixel_values, normalize_pixel_values};

pub struct HoughParams {
    /// Disk radius to search for.
    disk_radius: u32,
    /// Threshold of pixel brightness (0.0 - black, 1.0 - white) from which to apply the transform.
    brightness_threshold: f32,
    /// Larger dimension of transform input image (the original image is scaled to it).
    work_image_size: u32,
    /// Size of the area in which to search for the transform peak.
    peak_search_area_size: u32
}

impl Default for HoughParams {
    fn default() -> HoughParams {
        HoughParams {
            disk_radius: 676,
            brightness_threshold: 0.75,
            work_image_size: 512,
            peak_search_area_size: 30,
        }
    }
}

/// Returns relative angle and scale between `image1` and `image2` and refined position of `center2`
/// that matches `center1`.
///
/// Images must be `PixelFormat::Mono32f`.
///
fn match_disk_center_angle_scale(
    image1: &Image,
    center1: &Vector2<f32>,
    image2: &Image,
    center2: &Vector2<i32>,
    ring: &comp::RingMask,
    logger: &Logger
) -> (f32, f32, Vector2<f32>) {
    const COARSE_ANGLE_STEP: f32 = 3.0 * std::f32::consts::PI / 180.0;
    const ANGLE_REFINEMENT_RANGE: f32 = 2.0 * std::f32::consts::PI / 180.0;
    const ANGLE_REFINEMENT_STEP: f32 = 0.02 * std::f32::consts::PI / 180.0;

    const SCALE_REFINEMENT_RANGE: f32 = 0.01;
    // images are about 2500 pixels wide; this step amounts to max. 0.25 pixel of movement for aligned images
    const SCALE_REFINEMENT_STEP: f32 = 0.0002;

    let mut scale = 1.0;

    const TRANSLATION_REFINEMENT_STEP: f32 = 0.1;

    let center1_rounded = comp::to_f32(&comp::to_i32(&center1));
    let image1_fract_translated = comp::translate_pixels::<f32, _>(
        image1,
        ring,
        &comp::to_i32(center1),
        &(center1_rounded - center1)
    );
    let image1_mono8 = image1_fract_translated.convert_pix_fmt(PixelFormat::Mono8, None);

    let image2_mono8 = image2.convert_pix_fmt(PixelFormat::Mono8, None);

    // initial coarse search for the relative angle
    let (mut angle, _) = comp::find_relative_angle(
        &image1_mono8,
        &image2_mono8,
        &comp::to_i32(&center1),
        &comp::to_f32(center2),
        -180.0f32.to_radians(),
        180.0f32.to_radians(),
        COARSE_ANGLE_STEP,
        1.0,
        &ring
    );
    logger.verbose(&format!("initial angle estimation: {:.0}°", &angle.to_degrees()));

    let mut refined_center2 = comp::to_f32(&center2);

    // refinement: derotation + translation
    const MAX_REFINEMENT_ITERS: usize = 16;
    let mut iter = 0;
    let mut prev_angle = angle;
    loop {
        let (new_angle, rotated) = comp::find_relative_angle(
            &image1_mono8,
            &image2_mono8,
            &comp::to_i32(&center1),
            &refined_center2,
            angle - ANGLE_REFINEMENT_RANGE,
            angle + ANGLE_REFINEMENT_RANGE,
            ANGLE_REFINEMENT_STEP,
            1.0,
            ring
        );

        angle = new_angle;

        let translation = comp::find_translation_vector_subpixel2::<u8, _>(
            &image1_mono8,
            &rotated,
            &comp::to_i32(&center1),
            &refined_center2,
            ring,
            16.0,
            4.0,
            TRANSLATION_REFINEMENT_STEP,
            0.0
        ) - (center1_rounded - center1);

        refined_center2 += translation;

        logger.verbose(&format!(
            "refined angle: {:>6.2}°, translation: ({:>6.2}, {:>6.2})",
            &angle.to_degrees(), translation.x, translation.y
        ));

        iter += 1;
        if iter >= MAX_REFINEMENT_ITERS ||
            translation.x.abs() <= TRANSLATION_REFINEMENT_STEP &&
            translation.y.abs() <= TRANSLATION_REFINEMENT_STEP &&
            (prev_angle - angle).abs() < ANGLE_REFINEMENT_STEP {

            scale = comp::find_relative_scale(
                &image1_mono8,
                &rotated,
                &comp::to_i32(&center1),
                &refined_center2,
                1.0 - SCALE_REFINEMENT_RANGE,
                1.0 + SCALE_REFINEMENT_RANGE,
                SCALE_REFINEMENT_STEP,
                ring
            );
            logger.verbose(&format!("scale: {:.4}", scale));

            break;
        }

        prev_angle = angle;
    }

    (angle, scale, refined_center2)
}

/// Image must be `PixelFormat::Mono32f`.
fn estimate_disk_center_position(image: &Image, hough_params: &HoughParams) -> Vector2<i32> {
    let (image_small, factor) = comp::downscale(&image, hough_params.work_image_size);
    let hough_small = comp::hough_circle_accumulator(
        &image_small,
        hough_params.brightness_threshold,
        (factor * hough_params.disk_radius as f64) as u32
    );

    let middle_start = Vector2{
        x: (image_small.width() - hough_params.peak_search_area_size) as i32 / 2,
        y: (image_small.height() - hough_params.peak_search_area_size) as i32 / 2
    };
    let middle_frag = hough_small.fragment_copy(
        &comp::to_point(&middle_start),
        hough_params.peak_search_area_size,
        hough_params.peak_search_area_size,
        true
    );
    let mut disk_center = Vector2::<i32>::zero();
    let mut maxval = 0.0;
    for y in 0..middle_frag.height() {
        let line = middle_frag.line::<f32>(y);
        for x in 0..middle_frag.width() {
            if line[x as usize] > maxval {
                disk_center = Vector2{ x: x as i32, y: y as i32 };
                maxval = line[x as usize];
            }
        }
    }
    disk_center += middle_start;
    disk_center = comp::to_i32(&(comp::to_f32(&disk_center) / factor as f32));

    disk_center
}

/// Performs alignment (with derotation) of "per site" HDR images (stacks of all HDRs in each site); the images must be
/// preprocessed with NRGF and Sobel filter.
///
/// # Parameters
///
/// * `file_names` - Input file names in chronological order.
/// * `ring_r_inner` - Inner radius (in pixels) of comparison ring; should cover the stacked lunar disks.
/// * `ring_r_outer` - Outer radius (in pixels) of comparison ring; must fit within each image.
/// * `hough_params` - Parameters of circle Hough transform used to initially estimate disk center's position.
///
/// Returns (disk centers, angles relative to first image, scales relative to the first image).
///
pub fn align_per_site_hdr_images(
    file_names: &[String],
    ring_r_inner: i32,
    ring_r_outer: i32,
    hough_params: HoughParams,
    logger: &Logger
) -> (Vec<Vector2<f32>>, Vec<f32>, Vec<f32>) {
    let ring = comp::RingMask::new(ring_r_inner, ring_r_outer);

    let mut prev_image = Image::load(&file_names[0], image::FileType::Auto).unwrap();
    let mut prev_disk_center = comp::to_f32(&estimate_disk_center_position(&prev_image, &hough_params));

    let mut absolute_angles: Vec<f32> = vec![0.0];
    let mut disk_centers: Vec<Vector2<f32>> = vec![prev_disk_center];
    let mut absolute_scales: Vec<f32> = vec![1.0];

    let mut cumulative_angle = 0.0;
    let mut cumulative_scale = 1.0;

    for file_name in file_names.iter().skip(1) {
        logger.info(&format!("\n---------------\nLoading file {}", file_name));

        let image = Image::load(file_name, image::FileType::Auto).unwrap();
        let estimated_disk_center = estimate_disk_center_position(&image, &hough_params);

        let (angle, scale, actual_disk_center) = match_disk_center_angle_scale(
            &prev_image,
            &prev_disk_center,
            &image,
            &estimated_disk_center,
            &ring,
            logger
        );

        cumulative_angle += angle;
        cumulative_scale *= scale;

        absolute_angles.push(cumulative_angle);
        disk_centers.push(actual_disk_center);
        absolute_scales.push(cumulative_scale);

        prev_image = image;
        prev_disk_center = actual_disk_center;
    }

    (disk_centers, absolute_angles, absolute_scales)
}

/// Produces aligned images padded to the common bounding box.
///
/// When aligning images without rotation, give `ref_points[0]` as (0, 0), then the remaining elements are simply
/// translations relative to the first image.
///
/// # Parameters
///
/// * `image_file_names` - Input image file names.
/// * `angles` - Element [i] is the rotation angle (clockwise in left-handed coord. system) relative to the first image,
///     with `ref_points[i]` as the rotation center.
/// * `scales` - Element [i] is the scale factor relative to the first image, with `ref_points[i]`
///     as the scaling center.
/// * `ref_points` - Positions of the common reference point; used as rotation centers.
/// * `aligned_image_handler` - Function to call for each aligned image; parameters: source file name, image,
///     bounding box center.
/// * `averaged_image_handler` - Function to call for the averaged image.
/// * `exclude_moon_diameter` - If `Some`, pixels covered by the Moon are skipped when creating the averaged image.
/// * `background_threshold` - Used when `exclude_moon_diameter` is `Some`.
///
pub fn process_aligned_sequence<F1, F2>(
    image_file_names: &[String],
    angles: &[f32],
    scales: &[f32],
    ref_points: &[Vector2<f32>],
    aligned_image_handler: Option<F1>,
    averaged_image_handler: Option<F2>,
    logger: &Logger,
    exclude_moon_diameter: Option<u32>,
    background_threshold: f32
) where F1: Fn(&str, &Image, &Vector2<f32>),
        F2: Fn(&Image)
{
    // first, find the bounding box extents when the common reference point used for alignment is at (0, 0)
    let mut bbox_xmin = std::i32::MAX;
    let mut bbox_ymin = std::i32::MAX;
    let mut bbox_xmax = std::i32::MIN;
    let mut bbox_ymax = std::i32::MIN;

    logger.verbose("Checking input image dimensions...");
    let image_sizes: Vec<(u32, u32)> = image_file_names.iter().map(
        |fname| { let (w, h, _, _) = Image::image_metadata(fname, image::FileType::Auto).unwrap(); (w, h) }
    ).collect();

    for (i, img_size) in image_sizes.iter().enumerate() {
        let rot: Basis2<f32> = Rotation2::from_angle(Rad(angles[i]));

        let dc = comp::to_i32(&ref_points[i]);

        let upper_left   = Vector2{ x: -dc.x,                    y: -dc.y };
        let upper_right  = Vector2{ x: img_size.0 as i32 - dc.x, y: -dc.y };
        let bottom_left  = Vector2{ x: -dc.x,                    y: img_size.1 as i32 -dc.y };
        let bottom_right = Vector2{ x: img_size.0 as i32 -dc.x,  y: img_size.1 as i32 -dc.y };

        let corners = [upper_left, upper_right, bottom_left, bottom_right];
        let transformed_corners: Vec<Vector2<i32>> = corners.iter().map(
            |p| comp::to_i32(&(rot.rotate_vector(comp::to_f32(p)) / scales[i]))
        ).collect();

        bbox_xmin = bbox_xmin.min(transformed_corners.iter().min_by(|p1, p2| p1.x.cmp(&p2.x)).unwrap().x);
        bbox_xmax = bbox_xmax.max(transformed_corners.iter().max_by(|p1, p2| p1.x.cmp(&p2.x)).unwrap().x);
        bbox_ymin = bbox_ymin.min(transformed_corners.iter().min_by(|p1, p2| p1.y.cmp(&p2.y)).unwrap().y);
        bbox_ymax = bbox_ymax.max(transformed_corners.iter().max_by(|p1, p2| p1.y.cmp(&p2.y)).unwrap().y);
    }

    logger.verbose(&format!("Bounding box: {}x{} pixels.", (bbox_xmax - bbox_xmin + 1), (bbox_ymax - bbox_ymin + 1)));

    let mut sum_image = Image::new(
        (bbox_xmax - bbox_xmin + 1) as u32,
        (bbox_ymax - bbox_ymin + 1) as u32,
        None,
        PixelFormat::Mono32f,
        None,
        true
    );

    let mut sum_pixel_counters = vec![0usize; (sum_image.width() * sum_image.height()) as usize];

    // second, put each image in the bounding box using its translation, rotation and scaling and save the result
    for (i, file_name) in image_file_names.iter().enumerate() {
        logger.info(&format!("\nprocessing: {}", file_name));

        let rot: Basis2<f32> = Rotation2::from_angle(Rad(-angles[i] as f32));

        let refp = &ref_points[i];

        let bb_center = Vector2{ x: -bbox_xmin as f32, y: -bbox_ymin as f32 };

        let src_img = replace_nans(Image::load(file_name, image::FileType::Auto).unwrap());

        let mut aligned_img = Image::new(
            (bbox_xmax - bbox_xmin + 1) as u32,
            (bbox_ymax - bbox_ymin + 1) as u32,
            None,
            PixelFormat::Mono32f,
            None,
            true
        );

        let src_values_per_line = src_img.values_per_line::<f32>();
        let src_pixels = src_img.pixels::<f32>();

        let aligned_width = aligned_img.width() as i32;
        let aligned_height = aligned_img.height() as i32;

        for y in 0..aligned_height {
            let dest_line = aligned_img.line_mut::<f32>(y as u32);
            for x in 0..aligned_width {
                // perform bilinear interpolation in `src_img`
                let p_src = rot.rotate_vector(Vector2{ x: x as f32, y: y as f32 } - bb_center) * scales[i] + refp;

                let Vector2{ x: src_x, y: src_y } = p_src;

                let src_x_lo = src_x.floor() as i32;
                let src_y_lo = src_y.floor() as i32;

                if src_x_lo >= 0 && src_x_lo < src_img.width() as i32 - 1 &&
                   src_y_lo >= 0 && src_y_lo < src_img.height() as i32 - 1 {

                    let src_x_lo = src_x_lo as usize;
                    let src_y_lo = src_y_lo as usize;

                    let tx = src_x.fract() as f32;
                    let ty = src_y.fract() as f32;

                    let v_00 = src_pixels[src_x_lo +     src_y_lo * src_values_per_line];
                    let v_10 = src_pixels[src_x_lo + 1 + src_y_lo * src_values_per_line];
                    let v_11 = src_pixels[src_x_lo + 1 + (src_y_lo + 1) * src_values_per_line];
                    let v_01 = src_pixels[src_x_lo     + (src_y_lo + 1) * src_values_per_line];

                    dest_line[x as usize] =
                        v_00 * (1.0 - tx) * (1.0 - ty) +
                        v_10 * tx * (1.0 - ty) +
                        v_11 * tx * ty +
                        v_01 * (1.0 - tx) * ty;
                }
            }
        }

        if averaged_image_handler.is_some() {
            let lunar_disk: Option<comp::Disk> = if let Some(d_moon) = exclude_moon_diameter {
                Some(comp::Disk{
                    center: comp::find_lunar_disk_center(&aligned_img, d_moon as f32 / 2.0, background_threshold),
                    radius: d_moon / 2
                })
            } else {
                None
            };

            comp::add_image(&mut sum_image, &mut sum_pixel_counters, &aligned_img, Some(0.0), lunar_disk);
        }

        match &aligned_image_handler {
            Some(handler) => handler(file_name, &aligned_img, &bb_center),
            _ => ()
        }
    }

    if let Some(handler) = averaged_image_handler {
        let w = sum_image.width();
        let h = sum_image.height();
        for y in 0..h {
            let line = sum_image.line_mut::<f32>(y);
            for x in 0..w {
                let num_pixels_added = sum_pixel_counters[(x + y * w) as usize];
                if num_pixels_added > 0 {
                    line[x as usize] /= num_pixels_added as f32;
                }
            }
        }

        handler(&sum_image);
    }
}

/// Replaces all NaN values with the largest non-NaN value.
#[must_use]
pub fn replace_nans(mut image: Image) -> Image {
    assert!(image.pixel_format() == PixelFormat::Mono32f);
    let max = image.pixels::<f32>().iter().max_by(|a, b|
        if a.is_nan() { std::cmp::Ordering::Less } else if b.is_nan() { std::cmp::Ordering::Greater } else {
            a.partial_cmp(&b).unwrap()
        }
    ).unwrap().clone();
    for p in image.pixels_mut::<f32>() {
        if p.is_nan() { *p = max; }
    }

    image
}

fn load_for_alignment(file_name: &str) -> Image {
    replace_nans(Image::load(file_name, image::FileType::Auto).unwrap())
}

pub fn align_single_site_hdr_images(
    file_names: &[String],
    brightness_threshold: f32,
    ref_block_pos: &Vector2::<i32>,
    ref_block_size: u32,
    detrending_step: usize,
    logger: &Logger
) -> Vec<Vector2<f32>> {

    let mask = comp::NoDiskMask::new(0, ref_block_size as i32, ref_block_size as i32);

    logger.info(&format!("processing 1/{}: {}", file_names.len(), &file_names[0]));
    let mut prev_img = load_for_alignment(&file_names[0]);

    let mut do_not_detrend_from: Option<usize> = None;

    let mut translations = vec![Vector2{ x: 0.0, y: 0.0 }];

    for (i, file_name) in file_names.iter().enumerate().skip(1) {
        logger.info(&format!("\nprocessing {}/{}: {}", i + 1, file_names.len(), &file_name));
        let curr_img = load_for_alignment(file_name);

        if do_not_detrend_from.is_none() && curr_img
            .fragment_copy(&comp::to_point(&(ref_block_pos - Vector2{ x: ref_block_size as i32 / 2, y: ref_block_size as i32 / 2 })), ref_block_size, ref_block_size, true)
            .pixels::<f32>().iter().filter(|&p| !p.is_nan() && *p >= 100000.0).count() > 50 {
            do_not_detrend_from = Some(i);
        }

        let translation = comp::find_translation_vector_subpixel2::<f32, _>(
            &prev_img, &curr_img, &ref_block_pos, &comp::to_f32(ref_block_pos), &mask, 8.0, 2.0, 0.01, brightness_threshold
        );

        logger.info(&format!("translation: ({:.3}, {:.3})", translation.x, translation.y));
        translations.push(translation);

        prev_img = curr_img;
    }

    // Look for a trend.
    let mut cumulative_t = vec![translations[0]];
    for (i, t) in translations.iter().enumerate().skip(1) {
        cumulative_t.push(cumulative_t[i - 1] + t);
    }

    logger.verbose("Detrending to remove drift...");

    // Aligning on the prominence frame-to-frame results in a steadily drifting image.
    // Counter this by finding the translation between every `STEP` images and modifying
    // the translations between accordingly.

    let mut corrections = vec![Vector2::<f32>::zero()];
    let mut total_correction = Vector2::<f32>::zero();
    prev_img = load_for_alignment(&file_names[0]);
    let mut prev_cumulative = cumulative_t[0];
    let file_count = file_names.len();
    for i in (0..file_count).step_by(detrending_step).skip(1).chain(
        if (file_count - 1) % detrending_step == 0 { 1..=0 /*empty*/ } else { file_count - 1..=file_count - 1 /*last element*/ }
    ) {
       let curr_img = load_for_alignment(&file_names[i]);
       let translation = comp::find_translation_vector_subpixel2::<f32, _>(
           &prev_img, &curr_img, &ref_block_pos, &comp::to_f32(ref_block_pos), &mask, 20.0, 2.0, 0.01, brightness_threshold
       );

       let original = cumulative_t[i] - prev_cumulative;
       let delta = translation - original;
       logger.verbose(&format!("{}: found delta: ({:.4}, {:.4})", i, delta.x, delta.y));
       total_correction += delta;

       corrections.push(delta);

       prev_img = curr_img;
       prev_cumulative = cumulative_t[i];
    }

    let mut detrended_translations = vec![translations[0]];
    for i in (0..translations.len()).step_by(detrending_step) {
        let correction_chunk_len = (i + detrending_step).min(translations.len() - 1) - i;
        if correction_chunk_len == 0 {
            break;
        }

        let correction = corrections[i / detrending_step + 1];

        let delta = correction / correction_chunk_len as f32;
        for j in i + 1 .. i + 1 + correction_chunk_len {
             if do_not_detrend_from.is_none() || j < do_not_detrend_from.unwrap() {
                 detrended_translations.push(translations[j] + delta);
             } else {
                detrended_translations.push(translations[j]);
             }
        }
    }

    detrended_translations
}

// Linear (for now) interpolation of all `None` values.
#[must_use]
pub fn interpolate<T, F: Fn(&T) -> f32>(values: &Vec<Option<T>>, value_getter: F) -> Vec<f32> {
    // list of non-empty input values and their indices
    let input_values: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_some())
        .map(|(i, v)| (i, value_getter(v.as_ref().unwrap())))
        .collect();

    let mut result = vec![];
    for i in 0..input_values.len() - 1 {
        let delta = input_values[i + 1].1 - input_values[i].1;
        let count = input_values[i + 1].0 - input_values[i].0;
        for j in 0..count {
            result.push(input_values[i].1 + j as f32 * delta / count as f32);
        }
    }
    result.push(input_values.last().unwrap().1);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_2_values_interpolate() {
        let values: Vec<Option<f32>> = vec![Some(0.0), None, None, Some(3.0)];
        let interpolated = interpolate(&values, |v| *v);
        assert_eq!(
            vec![0.0, 1.0, 2.0, 3.0],
            interpolated
        );
    }

    #[test]
    fn given_3_values_interpolate() {
        let values: Vec<Option<f32>> = vec![Some(0.0), None, Some(2.0), None, None, None, Some(-2.0)];
        let interpolated = interpolate(&values, |v| *v);
        assert_eq!(
            vec![0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            interpolated
        );
    }

}
