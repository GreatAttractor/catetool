//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Entry point and main functions of the `catetool` executable.
//!

mod align;
mod args;
mod image;
mod logging;
mod utils;

use args::{InputFiles, ModeOfOperation};
use image::{Image};
use logging::Logger;
use std::io::prelude::*;
use std::path::Path;

const VERSION_STRING: &'static str = include_str!(concat!(env!("OUT_DIR"), "/version"));

fn print_header() {
    println!(r#"
_________________

   {}
   Image alignment for the Continental-America Telescopic Eclipse Experiment

   Copyright © 2020 Filip Szczerek <ga.software@yahoo.com>

   This program is licensed under MIT license (see LICENSE.txt for details).

_________________
"#,
        VERSION_STRING
    );
}

fn mode_align_single_site(config: args::Configuration, logger: &Logger) {
    let output_dir = match config.output_dir().clone() {
        Some(output_dir) => output_dir,
        None => ".".to_string()
    };
    let output_avg_file = match config.output_avg_file().clone() {
        Some(output_avg_file) => output_avg_file,
        None => Path::new(&output_dir).join("averaged.fits").to_str().unwrap().to_string()
    };
    let ref_block_pos = config.ref_block_position().unwrap();
    let save_aligned_single_site = config.save_aligned();
    let exclude_moon_diameter = config.exclude_moon_diameter();
    let background_threshold = config.background_threshold();
    let input_files: Vec<String> = match config.take_input_files() {
        InputFiles::CommandLineList(list) => list,
        InputFiles::ListFile(list_file) => utils::parse_list_file(&list_file)
    };

    const REF_BLOCK_SIZE: u32 = 150;
    const THRESHOLD: f32 = 15000.0;

    let translations = align::align_single_site_hdr_images(
        &input_files, THRESHOLD, &ref_block_pos, REF_BLOCK_SIZE, &logger
    );
    let angles: Vec<f32> = (0..translations.len()).into_iter().map(|_| 0.0).collect(); // all zeros
    let scales: Vec<f32> = (0..translations.len()).into_iter().map(|_| 1.0).collect(); // all ones
    let mut cumulative_t = vec![translations[0]];
    for (i, t) in translations.iter().enumerate().skip(1) {
        cumulative_t.push(cumulative_t[i - 1] + t);
    }

    {
        let diag_file_name = Path::new(&output_dir).join("diagnostic_data.txt").to_str().unwrap().to_string();
        let mut diag_file = std::fs::OpenOptions::new().read(false).write(true).create(true).truncate(true)
            .open(diag_file_name).unwrap();
        write!(diag_file, "{}\n\n", VERSION_STRING).unwrap();
        write!(diag_file, "Files and shifts (relative to the first file).\n\n").unwrap();
        write!(diag_file, "BEGIN LIST\n").unwrap();
        for i in 0..input_files.len() {
            write!(diag_file, "{}\n{:.3}   {:.3}\n", input_files[i], cumulative_t[i].x, cumulative_t[i].y)
                .unwrap();
        }
    }

    logger.info("\nSaving aligned images...");
    align::process_aligned_sequence(
        &input_files,
        &angles,
        &scales,
        &cumulative_t,
        Some(|src_file_name: &str, image: &Image, bbox_center: &cgmath::Vector2<f32>| {
            if save_aligned_single_site {
                let output_file_name: String = Path::new(&output_dir).join(
                    Path::new(src_file_name).file_stem().unwrap().to_str().unwrap().to_string() + "_aligned.fits"
                ).to_str().unwrap().to_string();
                logger.verbose(&format!("saving: {}", output_file_name));
                image.save(&output_file_name, image::FileType::Fits).unwrap();
            }

            // save the aligned reference block for verification
            let ref_block_file_name = Path::new(&output_dir).join(
                Path::new(src_file_name).file_stem().unwrap().to_str().unwrap().to_string()
                + "_aligned_ref_block.bmp"
            ).to_str().unwrap().to_string();

            align::replace_nans(align::scale_and_clip_pixel_values(
                image.fragment_copy(
                    &align::to_point(&(ref_block_pos + align::to_i32(bbox_center) - cgmath::Vector2{
                        x: REF_BLOCK_SIZE as i32 / 2,
                        y: REF_BLOCK_SIZE as i32 / 2
                    })),
                    REF_BLOCK_SIZE, REF_BLOCK_SIZE, true
                ),
                0.00002, 1.0
            )).convert_pix_fmt(image::PixelFormat::Mono8, None)
              .save(&ref_block_file_name, image::FileType::Bmp).unwrap();
        }),
        Some(|image: &Image| {
            logger.info(&format!("\nsaving: {}", output_avg_file));
            image.save(&output_avg_file, image::FileType::Fits).unwrap();
        }),
        &logger,
        exclude_moon_diameter,
        background_threshold
    );
}

fn mode_use_precalc_values(config: args::Configuration, logger: &Logger) {
    let output_dir = match config.output_dir().clone() {
        Some(output_dir) => output_dir,
        None => ".".to_string()
    };
    let output_avg_file = match config.output_avg_file().clone() {
        Some(output_avg_file) => output_avg_file,
        None => Path::new(&output_dir).join("averaged.fits").to_str().unwrap().to_string()
    };
    let save_aligned = config.save_aligned();
    let exclude_moon_diameter = config.exclude_moon_diameter();
    let background_threshold = config.background_threshold();
    let list_file_name: String = match config.take_input_files() {
        InputFiles::ListFile(list_file) => list_file,
        _ => panic!("Expected list file.")
    };

    let (input_files, input_ref_positions, input_angles, input_scales) =
        utils::parse_list_file_for_precalc_mode(&list_file_name);

    if !input_ref_positions[0].is_some() || !input_ref_positions.last().unwrap().is_some() {
        panic!("First and last position must be specified.");
    }

    let interpolated_pos_x = align::interpolate(&input_ref_positions, |v| v.x);
    let interpolated_pos_y = align::interpolate(&input_ref_positions, |v| v.y);
    let interpolated_angles = align::interpolate(&input_angles, |a| *a);
    let interpolated_scales = align::interpolate(&input_scales, |a| *a);

    let ref_pos: Vec<cgmath::Vector2<f32>> = interpolated_pos_x.iter().zip(interpolated_pos_y.iter())
        .map(|xy| cgmath::Vector2{ x: *xy.0, y: *xy.1 })
        .collect();

    logger.info("\nProcessing aligned images...");
    align::process_aligned_sequence(
        &input_files,
        &interpolated_angles,
        &interpolated_scales,
        &ref_pos,
        Some(|src_file_name: &str, image: &Image, _: &cgmath::Vector2<f32>| {
            if save_aligned {
                let output_file_name: String = Path::new(&output_dir).join(
                    Path::new(src_file_name).file_stem().unwrap().to_str().unwrap().to_string() + "_aligned.fits"
                ).to_str().unwrap().to_string();
                logger.verbose(&format!("saving: {}", output_file_name));
                image.save(&output_file_name, image::FileType::Fits).unwrap();

                //TESTING #################
                align::normalize_pixel_values(image.clone(), 100.0).convert_pix_fmt(image::PixelFormat::Mono8, None)
                     .save(&(output_file_name + ".bmp"), image::FileType::Bmp).unwrap();
            }
        }),
        Some(|image: &Image| {
            logger.info(&format!("\nsaving: {}", output_avg_file));
            image.save(&output_avg_file, image::FileType::Fits).unwrap();
        }),
        &logger,
        exclude_moon_diameter,
        background_threshold
    );
}

fn mode_align_multiple_sites(config: args::Configuration, logger: &Logger) {
    let output_dir = match config.output_dir().clone() {
        Some(output_dir) => output_dir,
        None => ".".to_string()
    };
    let background_threshold = config.background_threshold();
    let save_aligned = config.save_aligned();
    let input_files: Vec<String> = match config.take_input_files() {
        InputFiles::CommandLineList(list) => list,
        InputFiles::ListFile(list_file) => utils::parse_list_file(&list_file)
    };

    let (disk_centers, angles, scales) = align::align_per_site_hdr_images(
        &input_files,
        730,
        903,
        align::HoughParams::default(),
        &logger
    );

    {
        let diag_file_name = Path::new(&output_dir).join("diagnostic_data.txt").to_str().unwrap().to_string();
        let mut diag_file = std::fs::OpenOptions::new().read(false).write(true).create(true).truncate(true)
            .open(diag_file_name).unwrap();
        write!(diag_file, "{}\n\n", VERSION_STRING).unwrap();
        write!(diag_file, "Reference points, rotations (deg.) and scale factors (relative to the first file).\n\n").unwrap();
        write!(diag_file, "BEGIN LIST\n").unwrap();
        for i in 0..input_files.len() {
            write!(
                diag_file,
                "{}\n{:.3}   {:.3}   {:.4}   {:.4}\n",
                input_files[i], disk_centers[i].x, disk_centers[i].y, angles[i].to_degrees(), scales[i]
            ).unwrap();
        }
    }

    logger.info("\nSaving aligned images...");
    align::process_aligned_sequence(
        &input_files,
        &angles,
        &scales,
        &disk_centers,
        Some(|src_file_name: &str, image: &Image, _: &cgmath::Vector2<f32>| {
            if save_aligned {
                let output_file_name: String = Path::new(&output_dir).join(
                    Path::new(src_file_name).file_stem().unwrap().to_str().unwrap().to_string() + "_aligned.fits"
                ).to_str().unwrap().to_string();
                logger.verbose(&format!("saving: {}", output_file_name));
                image.save(&output_file_name, image::FileType::Fits).unwrap();

                // TESTING #############
                align::scale_and_clip_pixel_values(image.clone(), 1.0, 1.0).convert_pix_fmt(image::PixelFormat::Mono8, None)
                     .save(&(output_file_name + ".bmp"), image::FileType::Bmp).unwrap();
            }
        }),
        None::<fn(_: &Image)>,
        &logger,
        None,
        background_threshold
    );
}

fn run_program() -> bool {
    print_header();
    println!();

    let config = match args::parse_command_line(std::env::args()) {
        Ok(config) => match config {
            None => return true, // help was requested
            Some(config) => config
        },
        Err(_) => { println!("\nUse --{} for more information.\n", args::cmdline::HELP); return false; }
    };

    let mode = config.mode();

    let logger = Logger::new(config.log_level());

    let tstart = std::time::Instant::now();

    match mode {
        ModeOfOperation::AlignSingleSite => mode_align_single_site(config, &logger),

        ModeOfOperation::AlignMultipleSites => mode_align_multiple_sites(config, &logger),

        ModeOfOperation::PrecalculatedValues => mode_use_precalc_values(config, &logger)
    }

    let elapsed = tstart.elapsed();
    let mins = elapsed.as_secs() / 60;
    let secs = elapsed.as_secs() % 60;
    let frac_secs = elapsed.as_secs_f32() - (mins * 60) as f32 - secs as f32;
    logger.info(&format!("Completed in {} min {:02}.{:0.0} s.", mins, secs, frac_secs * 10.0));

    true
}

fn main() {
    std::process::exit(if run_program() { 0 } else { 1 });
}
