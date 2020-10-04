//
// catetool - Image alignment for the Continental-America Telescopic Eclipse Experiment
// Copyright (c) 2020 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//

//!
//! Point and rectangle structs and operations.
//!

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg};
use num_traits::{FromPrimitive, ToPrimitive};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Point {
    pub x: i32,
    pub y: i32
}

impl Add for Point {
    type Output = Point;

    fn add(mut self, other: Point) -> Point {
        self += other;
        self
    }
}

impl Sub for Point {
    type Output = Point;

    fn sub(mut self, other: Point) -> Point {
        self -= other;
        self
    }
}

impl SubAssign for Point {
    fn sub_assign(&mut self, other: Point) {
        *self = Point {
            x: self.x - other.x,
            y: self.y - other.y,
        };
    }
}

impl Neg for Point {
    type Output = Point;

    fn neg(self) -> Point {
        Point {
            x: -self.x,
            y: -self.y
        }
    }
}


impl AddAssign for Point {
    fn add_assign(&mut self, other: Point) {
        *self = Point {
            x: self.x + other.x,
            y: self.y + other.y,
        };
    }
}

impl<T> Mul<T> for Point
where T: Copy + Mul<Output=T> + ToPrimitive + FromPrimitive {
    type Output = Point;

    fn mul(mut self, rhs: T) -> Point {
        self *= rhs;
        self
    }
}

impl<T> MulAssign<T> for Point
where T: Copy + Mul<Output=T> + ToPrimitive + FromPrimitive {
    fn mul_assign(&mut self, rhs: T) {
        self.x = (T::from_i32(self.x).unwrap() * rhs).to_i32().unwrap();
        self.y = (T::from_i32(self.y).unwrap() * rhs).to_i32().unwrap();
    }
}

impl<T> Div<T> for Point
where T: Copy + Div<Output=T> + ToPrimitive + FromPrimitive {
    type Output = Point;

    fn div(mut self, rhs: T) -> Point {
        self /= rhs;
        self
    }
}

impl<T> DivAssign<T> for Point
where T: Copy + Div<Output=T> + ToPrimitive + FromPrimitive {
    fn div_assign(&mut self, rhs: T) {
        self.x = (T::from_i32(self.x).unwrap() / rhs).to_i32().unwrap();
        self.y = (T::from_i32(self.y).unwrap() / rhs).to_i32().unwrap();
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Point {
    pub fn new(x: i32, y: i32) -> Point { Point{ x, y} }

    pub fn sqr_dist(&self) -> i32 {
        self.x.pow(2) + self.y.pow(2)
    }

    /// Cross-product.
    pub fn cross(&self, rhs: &Point) -> i32 {
        self.x * rhs.y - self.y * rhs.x
    }

    pub fn zero() -> Point { Point{ x: 0, y: 0 } }
}

#[derive(Copy, Clone, Default, PartialEq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32
}

impl Rect {
    pub fn contains_point(&self, p: &Point) -> bool {
        p.x >= self.x && p.x < self.x + self.width as i32 && p.y >= self.y && p.y < self.y + self.height as i32
    }


    pub fn contains_rect(&self, other: &Rect) -> bool {
        self.contains_point(&Point{ x: other.x, y: other.y }) &&
        self.contains_point(&Point{ x: other.x + other.width as i32 - 1,
                                    y: other.y + other.height as i32 - 1 })
    }


    pub fn get_pos(&self) -> Point { Point{ x: self.x, y: self.y } }

    pub fn inflate(&self, margin: i32) -> Rect {
        Rect{
            x: self.x - margin,
            y: self.y - margin,
            width: (self.width as i32 + 2 * margin) as u32,
            height: (self.height as i32 + 2 * margin) as u32
        }
    }
}
