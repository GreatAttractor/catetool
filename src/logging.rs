//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Logger struct.
//!

#[derive(Copy, Clone, Debug, strum_macros::EnumIter, PartialEq)]
pub enum Level {
    Quiet,
    Info,
    Verbose
}

pub struct Logger {
    level: Level
}

impl Logger {
    pub fn new(level: Level) -> Logger { Logger{ level } }

    pub fn info(&self, msg: &str) {
        if self.level as i32 >= Level::Info as i32 {
            println!("{}", msg);
        }
    }

    pub fn verbose(&self, msg: &str) {
        if self.level as i32 >= Level::Verbose as i32 {
            println!("{}", msg);
        }
    }
}
