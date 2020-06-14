//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Utilities.
//!

use cgmath;
use std::io::BufRead;
use std::path::Path;

/// Returns all lines of the specified file.
pub fn parse_list_file(file_name: &str) -> Vec<String> {
    let list_file = std::fs::OpenOptions::new().read(true).write(false).open(file_name).unwrap();
    let list_lines = std::io::BufReader::new(list_file).lines();

    list_lines.map(|line| line.unwrap()).collect()
}

/// Returns (input files, translations, angles, scales) read from `file_name`.
///
/// The specified file must be structured as follows:
///
///   ...                                    /
///   ...ignored...                          /
///   ...                                    /
///   BEGIN LIST                             /  everything before BEGIN LIST is ignored
///   file_name1                             /
///   refpos1_x refpos1_y angle1 scale1      /  reference position: translation, center of rotation and scaling
///   file_name2                             /
///   refpos2_x refpos2_y                    /  angle and scale are optional (will be assumed 0.0 and 1.0)
///   ...                                    /
///   file_name_n                            /  specifying ref. pos, angle and scale is optional; if not given,
///   file_name_n+1                          /  will be interpolated from neighboring files
///   file_name_n+2                          /
///   refpos_n+2_x refpos_n+2_y angle_n+2    /
///   ...                                    /
///
///
pub fn parse_list_file_for_precalc_mode(file_name: &str)
-> (Vec<String>, Vec<Option<cgmath::Vector2<f32>>>, Vec<Option<f32>>, Vec<Option<f32>>) {
    let list_file = std::fs::OpenOptions::new().read(true).write(false).open(file_name).unwrap();
    let list_lines = std::io::BufReader::new(list_file).lines();
    parse_list_file_for_precalc_mode_priv(list_lines, |f| Path::new(f).exists())
}

/// Implements `parse_list_file_for_precalc_mode`.
fn parse_list_file_for_precalc_mode_priv<B, F>(lines: std::io::Lines<B>, path_exists: F)
-> (Vec<String>, Vec<Option<cgmath::Vector2<f32>>>, Vec<Option<f32>>, Vec<Option<f32>>)
where B: std::io::BufRead,
      F: Fn(&str) -> bool
{
    let mut input_files: Vec<String> = vec![];
    let mut input_ref_positions: Vec<Option<cgmath::Vector2<f32>>> = vec![];
    let mut input_angles: Vec<Option<f32>> = vec![];
    let mut input_scales: Vec<Option<f32>> = vec![];

    let mut file_required = true;
    let mut add_empty_pos = false;
    for (i, line) in lines.enumerate().skip_while(|(_, line)| line.as_ref().unwrap() != "BEGIN LIST").skip(1) {
        dbg!(i, file_required, add_empty_pos);

        let line = line.unwrap();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() == 0 {
            panic!("Unexpected empty line {}.", i + 1);
        } else if path_exists(&line) {
            if add_empty_pos {
                input_ref_positions.push(None);
                input_angles.push(None);
                input_scales.push(None);
            }
            input_files.push(line);
            file_required = false;
            add_empty_pos = true;
            println!("added existing file");
        } else if file_required && !path_exists(&line) {
            panic!("Expected an existing file in line {}.", i + 1);
        } else if tokens.len() > 4 {
            panic!("Too many values ({}) in line {}.", tokens.len(), i + 1);
        } else {
            let x = tokens[0].parse::<f32>().unwrap();
            let y = tokens[1].parse::<f32>().unwrap();
            let angle = if tokens.len() > 2 { tokens[2].parse::<f32>().unwrap().to_radians() } else { 0.0 };
            let scale = if tokens.len() > 3 { tokens[3].parse::<f32>().unwrap() } else { 1.0 };
            if scale <= 0.0 {
                panic!("Scale value cannot be negative: {}.", scale);
            }
            input_ref_positions.push(Some(cgmath::Vector2{ x, y }));
            input_angles.push(Some(angle));
            input_scales.push(Some(scale));
            file_required = true;
            add_empty_pos = false;
            println!("parsed floats]");
        }
    }

    (input_files, input_ref_positions, input_angles, input_scales)
}

#[cfg(test)]
mod tests {
    use cgmath::Vector2;
    use super::parse_list_file_for_precalc_mode_priv;
    use std::io::BufRead;

    fn f_exists(file_name: &str) -> bool { file_name.starts_with("file") }

    #[test]
    fn given_existing_files_succeed() {
        let input =
r#"some ignored text
some ignored text
BEGIN LIST
file1
0 0 0 1
file2
0 0 0 1"#;

        let _ = parse_list_file_for_precalc_mode_priv(input.as_bytes().lines(), f_exists);
    }

    #[test]
    #[should_panic]
    fn given_nonexistent_files_fail() {
        let input =
r#"some ignored text

some ignored text
BEGIN LIST
file1
0 0 0 1
nonexistent
0 0 0 1"#;

        let _ = parse_list_file_for_precalc_mode_priv(input.as_bytes().lines(), f_exists);
    }

    #[test]
    fn when_all_values_provided_succeed() {
        let input =
r#"BEGIN LIST
file1
-0.5 2 3.0 0.9
file2
1 2 3 4
"#;

        let (files, refpos, angles, scales) = parse_list_file_for_precalc_mode_priv(input.as_bytes().lines(), f_exists);

        assert_eq!(vec!["file1", "file2"], files);

        assert_eq!(vec![
            Some(Vector2{ x: -0.5, y: 2.0 }),
            Some(Vector2{ x: 1.0, y: 2.0 })
        ], refpos);

        assert_eq!(vec![
            Some(3.0f32.to_radians()),
            Some(3.0f32.to_radians())
        ], angles);

        assert_eq!(vec![
            Some(0.9),
            Some(4.0)
        ], scales);
    }

    #[test]
    fn when_some_values_provided_succeed() {
        let input =
r#"BEGIN LIST
file1
-0.5 2 3.0 0.9
file2
1 2 3 4
file3
file4
7.0 8.0 5.0
file5
9.0 10.0
"#;

        let (files, refpos, angles, scales) = parse_list_file_for_precalc_mode_priv(input.as_bytes().lines(), f_exists);

        assert_eq!(vec!["file1", "file2", "file3", "file4", "file5"], files);

        assert_eq!(vec![
            Some(Vector2{ x: -0.5, y: 2.0 }),
            Some(Vector2{ x: 1.0, y: 2.0 }),
            None,
            Some(Vector2{ x: 7.0, y: 8.0 }),
            Some(Vector2{ x: 9.0, y: 10.0 })
        ], refpos);

        assert_eq!(vec![
            Some(3.0f32.to_radians()),
            Some(3.0f32.to_radians()),
            None,
            Some(5.0f32.to_radians()),
            Some(0.0)
        ], angles);

        assert_eq!(vec![
            Some(0.9),
            Some(4.0),
            None,
            Some(1.0),
            Some(1.0)
        ], scales);
    }

    #[test]
    #[should_panic]
    fn when_expected_file_missing_fail() {
        let input =
r#"BEGIN LIST
file1
1 2 3 4
file2
1 2 3 4
1 2 3 4"#;

        let _ = parse_list_file_for_precalc_mode_priv(input.as_bytes().lines(), f_exists);
    }

    #[test]
    #[should_panic]
    fn when_too_many_values_fail() {
        let input =
r#"BEGIN LIST
file1
1 2 3 4
file2
1 2 3 4 5"#;

        let _ = parse_list_file_for_precalc_mode_priv(input.as_bytes().lines(), f_exists);
    }

}