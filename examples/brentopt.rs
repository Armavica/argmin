// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::brentopt::Brent;

/// Test function: `f(x) = exp(-x) - exp(5-x/2)`
/// xmin == 2 log(2 exp(-5))
/// xmin ~= -8.6137056388801093812
/// f(xmin) == -exp(10)/4
/// f(xmin) ~= -5506.6164487016791292
struct TestFunc {}

impl ArgminOp for TestFunc {
    // one dimensional problem, no vector needed
    type Param = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        Ok((-x).exp() - (5. - x / 2.).exp())
    }
}

fn main() {
    let cost = TestFunc {};
    let solver = Brent::new(-10., 10.);

    let res = Executor::new(cost, solver, f64::nan())
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()
        .unwrap();
    println!("Result of brent:\n{}", res);
}
