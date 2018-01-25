//! Optimizaton toolbox
//!
//! TODOs
//!
//! * Stopping criterions which can be stacked, also return the reason why a computation terminated
//! * keep track of cost function values
//! * count the number of cost function / gradient evaluations and return them
//! * redesign how lower and upper bound are dealt with. making them optional should be better.
#![recursion_limit = "1024"]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![warn(missing_docs)]
#[macro_use]
extern crate error_chain;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num;
extern crate rand;

use std::fmt::Debug;
use std::default::Default;
use num::{Float, FromPrimitive, Num, ToPrimitive};
use parameter::ArgminParameter;
use result::ArgminResult;
use problem::Problem;
use errors::*;

/// Trait for cost function values
/// TODO: Do this with trait aliases once they work in rust.
pub trait ArgminCostValue
    : Float + FromPrimitive + ToPrimitive + Num + Default + Clone + Debug + PartialOrd
    {
}
impl<T> ArgminCostValue for T
where
    T: Float + FromPrimitive + ToPrimitive + Num + Default + Clone + Debug + PartialOrd,
{
}

/// Trait every solve needs to implement (in the future)
pub trait ArgminSolver<'a> {
    /// Parameter vector
    // type A: ArgminParameter<Self::A>;
    type A: ArgminParameter;
    /// Cost value
    type B: ArgminCostValue;
    /// Hessian
    type C;

    /// Initializes the solver and sets the state to its initial state
    fn init(&mut self, &'a Problem<'a, Self::A, Self::B, Self::C>, &Self::A) -> Result<()>;

    /// Moves forward by a single iteration
    fn next_iter(&mut self) -> Result<ArgminResult<Self::A, Self::B>>;

    /// Run initialization and iterations at once
    fn run(
        &mut self,
        &'a Problem<'a, Self::A, Self::B, Self::C>,
        &Self::A,
    ) -> Result<ArgminResult<Self::A, Self::B>>;

    /// Handles the stopping criteria
    fn terminate(&self) -> bool;
}

/// Definition of the return type of the solvers
pub mod result;

/// Traits for implementing parameter vectors
pub mod parameter;

/// Problem formulation
pub mod problem;

/// A set of test functions like Rosenbrock's function and so on.
pub mod testfunctions;

/// Backtracking line search
pub mod backtracking;

/// Simulated Annealing
pub mod sa;

/// Gradient Descent
pub mod gradientdescent;

/// Nelder Mead method
pub mod neldermead;

/// Newton method
pub mod newton;

/// Errors using `error-chain`
mod errors;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
