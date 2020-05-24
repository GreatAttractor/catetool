//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Build script.
//!

use chrono::prelude::Utc;

fn main() {
    let output_dir = std::env::var("OUT_DIR").unwrap();
    let version_path = std::path::Path::new(&output_dir).join("version");

    let version_str = format!(
        "{} {} (commit {}, {} {}, built on {})",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        get_commit_hash(),
        std::env::consts::OS, std::env::consts::ARCH,
        Utc::now().format("%Y-%m-%d %H:%M UTC")
    );

    std::fs::write(version_path, version_str).unwrap();
}

fn get_commit_hash() -> String {
    let output = std::process::Command::new("git")
        .arg("log").arg("-1")
        .arg("--pretty=format:%h")
        .arg("--abbrev=8")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();

    if output.status.success() {
        String::from_utf8_lossy(&output.stdout).to_string()
    } else {
        "unspecified".to_string()
    }
}
