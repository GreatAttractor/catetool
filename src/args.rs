//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Command-line options definitions and parsing.
//!

use crate::logging;
use strum::IntoEnumIterator;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ModeOfOperation {
    AlignSingleSite,
    AlignMultipleSites,
    PrecalculatedValues
}

pub mod cmdline {
    pub const HELP:                  &str = "help";
    pub const MODE_OF_OPERATION:     &str = "mode";
    pub const INPUT_FILES:           &str = "input_files";
    pub const OUTPUT_DIRECTORY:      &str = "output_dir";
    pub const OUTPUT_AVG_FILE:       &str = "output_avg_file";
    pub const INPUT_LIST:            &str = "input_list";
    pub const REF_BLOCK_POS:         &str = "ref_pos";
    pub const SAVE_ALIGNED:          &str = "save_aligned";
    pub const EXCLUDE_MOON_DIAMETER: &str = "exclude_moon_diameter";
    pub const BACKGROUND_THRESHOLD:  &str = "background_threshold";
    pub const LOG_LEVEL:             &str = "log_level";
    pub const DETRENDING_STEP:       &str = "detrending_step";
    pub const BLK_MATCH_THRESHOLD:   &str = "blk_match_threshold";
}

#[derive(Debug)]
pub enum InputFiles {
    CommandLineList(Vec<String>),
    ListFile(String),
}

#[derive(Debug)]
pub struct Configuration {
    mode: ModeOfOperation,
    output_dir: Option<String>,
    output_avg_file: Option<String>,
    input_files: InputFiles,
    ref_block_position: Option<cgmath::Vector2::<i32>>,
    save_aligned: bool,
    exclude_moon_diameter: Option<u32>,
    background_threshold: f32,
    detrending_step: usize,
    blk_match_threshold: f32,
    log_level: logging::Level
}

impl Configuration {
    pub fn mode(&self) -> ModeOfOperation { self.mode }
    pub fn output_dir(&self) -> &Option<String> { &self.output_dir }
    pub fn output_avg_file(&self) -> &Option<String> { &self.output_avg_file }
    pub fn take_input_files(self) -> InputFiles { self.input_files }
    pub fn ref_block_position(&self) -> Option<cgmath::Vector2<i32>> { self.ref_block_position }
    pub fn save_aligned(&self) -> bool { self.save_aligned }
    pub fn exclude_moon_diameter(&self) -> Option<u32> { self.exclude_moon_diameter }
    pub fn background_threshold(&self) -> f32 { self.background_threshold }
    pub fn detrending_step(&self) -> usize { self.detrending_step }
    pub fn blk_match_threshold(&self) -> f32 { self.blk_match_threshold }
    pub fn log_level(&self) -> logging::Level { self.log_level }
}

impl From<ModeOfOperation> for &str {
    fn from(m: ModeOfOperation) -> &'static str {
        match m {
            ModeOfOperation::AlignSingleSite     => "single-site",
            ModeOfOperation::AlignMultipleSites  => "multi-site",
            ModeOfOperation::PrecalculatedValues => "precalc"
        }
    }
}

impl From<logging::Level> for &str {
    fn from(level: logging::Level) -> &'static str {
        match level {
            logging::Level::Quiet   => "quiet",
            logging::Level::Info    => "info",
            logging::Level::Verbose => "verbose"
        }
    }
}

impl std::str::FromStr for logging::Level {
    type Err = ();
    fn from_str(s: &str) -> Result<logging::Level, ()> {

        for level in logging::Level::iter() {
            if s == Into::<&str>::into(level) {
                return Ok(level);
            }
        }

        Err(())
    }
}


pub fn print_help() {
    println!(
r#"Command-line options:

  --{} <mode>

    Mode of operation. Possible values:

        {:11}    align HDR images from a single site
        {:11}    align averaged HDR images of multiple sites
        {:11}    produce aligned images according to precalculated translations and rotations

    Mode "{}" requires the usage of --{} option. The list file may contain any text in the beginning, followed by:

        BEGIN LIST
        <file1>
        <x1> <y1> <angle> <scale>
        <file2>
        <x2> <y2> <angle> <scale>
        ...

    The angle (deg.) applies with (x, y) as the rotation center and may be skipped (will be presumed 0.0). Likewise, `scale` uses (x, y) as the scaling center and may be skipped (will be presumed 1.0).
    The x, y must be specified at least for the first and the last file. Any values not specified will be linearly interpolated.


    --{} <file1 file2 ...>

      Input files in alignment order.


    --{} <file>

      File containing the list of input files in alignment order (one file per line).
      If used with --{} {}, the format is more complex (see --{}).


    --{} <directory>

      Output directory in which to save the aligned images and reference blocks.


    --{} <x> <y>

      Position of the center of the big prominence's base in the first input file. Required when mode = {}.


    --{} <file>

      Path and name of the averaged output file. Default: "averaged.fits".


    --{} <yes|no>

      Whether to save aligned files in {} or {} mode. Default: no for {}, yes for {}.


    --{} <moon diameter in pixels>

      If specified, pixels covered by the Moon are skipped when creating the averaged image.


    --{} <value>

      Specifies the brightness threshold of sky background used with {}. Default: 3800.0.


    --{} <{}|{}|{}>

      Chooses the amount of messages to print during processing.


    --{} <value>

      Sets brightness threshold for block matching. Default: 15000. Use lower values for sites with low image brightness.
      Valid only when mode = {}.


    --{} <value>

      Sets the search step of translation detrending. Default: 10. Use lower values for sites with low image brightness.
      Valid only when mode = {}.

"#,
        cmdline::MODE_OF_OPERATION,
        Into::<&str>::into(ModeOfOperation::AlignSingleSite),
        Into::<&str>::into(ModeOfOperation::AlignMultipleSites),
        Into::<&str>::into(ModeOfOperation::PrecalculatedValues),
        Into::<&str>::into(ModeOfOperation::PrecalculatedValues), cmdline::INPUT_LIST,

        cmdline::INPUT_FILES,

        cmdline::INPUT_LIST,
        cmdline::MODE_OF_OPERATION, Into::<&str>::into(ModeOfOperation::PrecalculatedValues), cmdline::MODE_OF_OPERATION,

        cmdline::OUTPUT_DIRECTORY,

        cmdline::REF_BLOCK_POS, Into::<&str>::into(ModeOfOperation::AlignSingleSite),

        cmdline::OUTPUT_AVG_FILE,

        cmdline::SAVE_ALIGNED,
        Into::<&str>::into(ModeOfOperation::AlignSingleSite),
        Into::<&str>::into(ModeOfOperation::PrecalculatedValues),
        Into::<&str>::into(ModeOfOperation::AlignSingleSite),
        Into::<&str>::into(ModeOfOperation::PrecalculatedValues),

        cmdline::EXCLUDE_MOON_DIAMETER,

        cmdline::BACKGROUND_THRESHOLD, cmdline::EXCLUDE_MOON_DIAMETER,

        cmdline::LOG_LEVEL,
        Into::<&str>::into(logging::Level::Quiet),
        Into::<&str>::into(logging::Level::Info),
        Into::<&str>::into(logging::Level::Verbose),

        cmdline::BLK_MATCH_THRESHOLD,
        Into::<&str>::into(ModeOfOperation::AlignSingleSite),

        cmdline::DETRENDING_STEP,
        Into::<&str>::into(ModeOfOperation::AlignSingleSite),
    );
}

/// Returns the value of a single-valued option of type `T`.
fn get_option_value<T: std::str::FromStr>(
    option: &str,
    option_values: &std::collections::HashMap::<String, Vec<String>>
) -> Result<Option<T>, ()> {
    match option_values.get(option) {
        None => Ok(None),
        Some(vals) => if vals.is_empty() {
            eprintln!("Value missing for option {}.", option);
            Err(())
        } else if vals.len() > 1 {
            eprintln!("Too many values for option {}.", option);
            Err(())
        } else {
            match vals[0].parse::<T>() {
                Ok(value) => Ok(Some(value)),
                Err(_) => {
                    eprintln!("Invalid value for option {}: {}.", option, vals[0]);
                    Err(())
                }
            }
        }
    }
}

/// Returns Ok(None) if help was requested.
pub fn parse_command_line<I: Iterator<Item=String>>(stream: I) -> Result<Option<Configuration>, ()> {
    let allowed_options = vec![
     cmdline::HELP,
     cmdline::MODE_OF_OPERATION,
     cmdline::INPUT_FILES,
     cmdline::OUTPUT_DIRECTORY,
     cmdline::OUTPUT_AVG_FILE,
     cmdline::INPUT_LIST,
     cmdline::REF_BLOCK_POS,
     cmdline::SAVE_ALIGNED,
     cmdline::EXCLUDE_MOON_DIAMETER,
     cmdline::BACKGROUND_THRESHOLD,
     cmdline::LOG_LEVEL,
     cmdline::DETRENDING_STEP,
     cmdline::BLK_MATCH_THRESHOLD
    ];

    // key: option name
    let mut option_values = std::collections::HashMap::<String, Vec<String>>::new();

    let mut current: Option<&mut Vec<String>> = None;

    for arg in stream.skip(1) /*skip the binary name*/ {
        if arg.starts_with("--") {
            match &arg[2..] {
                cmdline::HELP => { print_help(); return Ok(None); },
                x if !allowed_options.contains(&x) => {
                    eprintln!("Unknown command-line option: {}.", x); return Err(());
                },
                opt => current = Some(option_values.entry(opt.to_string()).or_insert(vec![])),
            }
        } else {
            if current.is_none() {
                eprintln!("Unexpected value: {}.", arg);
                return Err(());
            } else {
                (*(*current.as_mut().unwrap())).push(arg);
            }
        }
    }

    let opt_mode = option_values.get(cmdline::MODE_OF_OPERATION);
    if opt_mode.is_none() || opt_mode.unwrap().is_empty() {
        eprintln!("Mode not specified."); return Err(());
    }
    let mode = match &opt_mode.unwrap()[0] {
        x if x == Into::<&str>::into(ModeOfOperation::AlignSingleSite) => ModeOfOperation::AlignSingleSite,
        x if x == Into::<&str>::into(ModeOfOperation::AlignMultipleSites) => ModeOfOperation::AlignMultipleSites,
        x if x == Into::<&str>::into(ModeOfOperation::PrecalculatedValues) => ModeOfOperation::PrecalculatedValues,
        x => {
            eprintln!("Invalid mode of operation: {}. Expected one of: {}, {}, {}.",
                x,
                Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                Into::<&str>::into(ModeOfOperation::AlignMultipleSites),
                Into::<&str>::into(ModeOfOperation::PrecalculatedValues)
            );
            return Err(());
        }
    };

    let output_dir = match option_values.get(cmdline::OUTPUT_DIRECTORY) {
        None => None,
        Some(vals) => if vals.is_empty() { None } else { Some(vals[0].clone()) }
    };

    let output_avg_file = match option_values.get(cmdline::OUTPUT_AVG_FILE) {
        None => None,
        Some(vals) => if vals.is_empty() { None } else { Some(vals[0].clone()) }
    };

    if !option_values.get(cmdline::INPUT_FILES).unwrap_or(&vec![]).is_empty() &&
       !option_values.get(cmdline::INPUT_LIST).unwrap_or(&vec![]).is_empty() {
        eprintln!("Cannot use both a list file and input files given in the command line.");
        return Err(());
    }

    let input_files: InputFiles = match option_values.get(cmdline::INPUT_FILES) {
        Some(vals) => if vals.is_empty() {
            eprintln!("Input files not specified.");
            return Err(());
        } else {
            InputFiles::CommandLineList(vals.clone())
        },
        None => match option_values.get(cmdline::INPUT_LIST) {
            Some(vals) => if vals.is_empty() {
                eprintln!("Input files not specified.");
                return Err(())
            } else {
                 InputFiles::ListFile(vals[0].clone())
             },
            None => {
                eprintln!("Input files not specified.");
                return Err(())
            }
        }
    };

    let ref_block_position: Option<cgmath::Vector2<i32>> = match option_values.get(cmdline::REF_BLOCK_POS) {
        Some(vals) => if vals.is_empty() {
            eprintln!("Reference block position not specified. Expected: <x> <y>.");
            return Err(());
        } else {
            let parsed_x = vals[0].parse::<i32>();
            let parsed_y = vals[1].parse::<i32>();
            if parsed_x.is_err() || parsed_y.is_err() {
                eprintln!("Invalid reference block position: {} {}.", vals[0], vals[1]);
                return Err(());
            } else {
                Some(cgmath::Vector2{ x: parsed_x.unwrap(), y: parsed_y.unwrap() })
            }
        },
        None => None
    };

    if mode == ModeOfOperation::AlignSingleSite && ref_block_position.is_none() {
        eprintln!("Refence block position must be specified.");
        return Err(());
    }

    let save_aligned = match option_values.get(cmdline::SAVE_ALIGNED) {
        None => if mode == ModeOfOperation::AlignSingleSite { false } else { true },
        Some(vals) => if vals.is_empty() {
            eprintln!("Value missing for option {}.", cmdline::SAVE_ALIGNED);
            return Err(());
        } else {
            match &(*vals[0]) {
                "yes" => true,
                "no" => false,
                _ => {
                    eprintln!("Invalid value for option {}: {}.", cmdline::SAVE_ALIGNED, vals[0]);
                    return Err(());
                }
            }
        }
    };

    let exclude_moon_diameter = get_option_value::<u32>(cmdline::EXCLUDE_MOON_DIAMETER, &option_values)?;

    let background_threshold = {
        match get_option_value::<f32>(cmdline::BACKGROUND_THRESHOLD, &option_values) {
            Err(_) => return Err(()),
            Ok(Some(val)) => if exclude_moon_diameter.is_none() {
                eprintln!("Unexpected option {} ({} was not specified).",
                    cmdline::BACKGROUND_THRESHOLD, cmdline::EXCLUDE_MOON_DIAMETER
                );
                return Err(());
            } else {
                val
            },
            Ok(None) => 3800.0
        }
    };

    let detrending_step = {
        match get_option_value::<usize>(cmdline::DETRENDING_STEP, &option_values) {
            Err(_) => return Err(()),

            Ok(Some(val)) => if mode != ModeOfOperation::AlignSingleSite {
                eprintln!("Unexpected option {} (mode is not {}).",
                    cmdline::DETRENDING_STEP, Into::<&str>::into(ModeOfOperation::AlignSingleSite)
                );
                return Err(());
            } else {
                val
            },

            Ok(None) => 10
        }
    };

    let blk_match_threshold = {
        match get_option_value::<f32>(cmdline::BLK_MATCH_THRESHOLD, &option_values) {
            Err(_) => return Err(()),

            Ok(Some(val)) => if mode != ModeOfOperation::AlignSingleSite {
                eprintln!("Unexpected option {} (mode is not {}).",
                    cmdline::BLK_MATCH_THRESHOLD, Into::<&str>::into(ModeOfOperation::AlignSingleSite)
                );
                return Err(());
            } else {
                val
            },

            Ok(None) => 15000.0
        }
    };

    let log_level = get_option_value::<logging::Level>(cmdline::LOG_LEVEL, &option_values)?
        .unwrap_or(logging::Level::Info);

    Ok(Some(Configuration{
        mode,
        output_dir,
        output_avg_file,
        input_files,
        ref_block_position,
        save_aligned,
        exclude_moon_diameter,
        background_threshold,
        detrending_step,
        blk_match_threshold,
        log_level
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prepends "--".
    macro_rules! as_opt { ($e:expr) => { ("--".to_string() + &$e.to_string()).as_str() } }

    #[test]
    fn when_help_requested_succeed() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::HELP)
            ].iter().map(|s| s.to_string())
        );
        assert!(config.ok().unwrap().is_none());
    }

    #[test]
    fn when_no_mode_fail() {
        let config = parse_command_line(
            [
                "binary"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_no_files_option_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite)
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_files_option_empty_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_FILES)
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_list_file_empty_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_LIST)
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_list_file_and_input_files_both_given_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_LIST), "mylist.txt",
                as_opt!(cmdline::INPUT_FILES), "file1", "file2"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_list_file_option_empty_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_LIST)
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_input_files_given_succeed() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3",
                as_opt!(cmdline::REF_BLOCK_POS), "1", "2"
            ].iter().map(|s| s.to_string())
        );
        match config.ok().unwrap().unwrap().input_files {
            InputFiles::CommandLineList(_) => (),
            _ => panic!("Expected: list of files.")
        }
    }

    #[test]
    fn when_align_single_site_and_no_ref_pos_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_ref_block_pos_given_succeed() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3",
                as_opt!(cmdline::REF_BLOCK_POS), "1", "2"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.ok().unwrap().unwrap().ref_block_position.unwrap() == cgmath::Vector2{ x: 1, y: 2 });
    }

    #[test]
    fn when_unknown_option_fail() {
        let config = parse_command_line(
            [
                "binary",
                "--some_unknown_option"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_invalid_save_aligned_value_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3",
                as_opt!(cmdline::REF_BLOCK_POS), "1", "2",
                as_opt!(cmdline::SAVE_ALIGNED), "BAD"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn when_good_save_aligned_value_succeed() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3",
                as_opt!(cmdline::REF_BLOCK_POS), "1", "2",
                as_opt!(cmdline::SAVE_ALIGNED), "yes"
            ].iter().map(|s| s.to_string())
        );
        assert_eq!(true, config.unwrap().unwrap().save_aligned());

        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignSingleSite),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3",
                as_opt!(cmdline::REF_BLOCK_POS), "1", "2",
                as_opt!(cmdline::SAVE_ALIGNED), "no"
            ].iter().map(|s| s.to_string())
        );
        assert_eq!(false, config.unwrap().unwrap().save_aligned());
    }

    #[test]
    fn given_detrending_step_and_not_single_site_mode_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignMultipleSites),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3",
                as_opt!(cmdline::DETRENDING_STEP), "5"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }

    #[test]
    fn given_blk_match_threshold_and_not_single_site_mode_fail() {
        let config = parse_command_line(
            [
                "binary",
                as_opt!(cmdline::MODE_OF_OPERATION), Into::<&str>::into(ModeOfOperation::AlignMultipleSites),
                as_opt!(cmdline::INPUT_FILES), "file1", "file2", "file3",
                as_opt!(cmdline::BLK_MATCH_THRESHOLD), "10.0"
            ].iter().map(|s| s.to_string())
        );
        assert!(config.is_err());
    }
}
