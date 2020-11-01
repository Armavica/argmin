// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Brent's method
//!
//! A minimization algorithm combining parabolic interpolation and the
//! golden-section method.  It has the reliability of the golden-section
//! method, but can be faster thanks to the parabolic interpolation steps.
//!
//! # References:
//!
//! "An algorithm with guaranteed convergence for finding a minimum of
//! a function of one variable", _Algorithms for minimization without
//! derivatives_, Richard P. Brent, 1973, Prentice-Hall.

/// Implementation of Brent's optimization method,
use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// Brent's method
///
/// A minimization algorithm combining parabolic interpolation and the
/// golden-section method.  It has the reliability of the golden-section
/// method, but can be faster thanks to the parabolic interpolation steps.
///
/// # References:
///
/// "An algorithm with guaranteed convergence for finding a minimum of
/// a function of one variable", _Algorithms for minimization without
/// derivatives_, Richard P. Brent, 1973, Prentice-Hall.
#[derive(Clone, Serialize, Deserialize)]
pub struct Brent<F> {
    /// relative tolerance
    eps: F,
    /// absolute tolerance
    t: F,
    /// left boundary of current interval
    a: F,
    /// right boundary of current interval
    b: F,
    /// last point where f was evaluated
    u: F,
    /// previous value of w
    v: F,
    /// point with the current second lowest value of f
    w: F,
    /// point with the current lowest value of f
    x: F,
    /// value of f in v
    fv: F,
    /// value of f in w
    fw: F,
    /// value of f in x
    fx: F,
    /// value of p/q in the second last step
    e: F,
    /// value of p/q in the last step
    d: F,
    /// (3-sqrt(5)) / 2
    c: F,
}

impl<F: ArgminFloat> Brent<F> {
    /// Constructor
    ///
    /// The values `min` and `max` must bracket the minimum of the function.
    pub fn new(min: F, max: F) -> Brent<F> {
        Brent {
            eps: F::epsilon().sqrt(),
            t: F::from_f64(1e-5).unwrap(),
            a: min,
            b: max,
            u: F::nan(),
            v: F::nan(),
            w: F::nan(),
            x: F::nan(),
            fv: F::nan(),
            fw: F::nan(),
            fx: F::nan(),
            e: F::zero(),
            d: F::zero(),
            c: F::from_f64((3f64 - 5f64.sqrt()) / 2f64).unwrap(),
        }
    }

    /// Set the tolerance to the value required.
    ///
    /// The algorithm will return an approximation `x` of a local
    /// minimum of the function, with an accuracy smaller than `3 tol`,
    /// where `tol = eps*abs(x) + t`.
    /// It is useless to set `eps` to less than the square root of the
    /// machine precision (`F::epsilon().sqrt()`), which is its default
    /// value.  The default value of `t` is `1e-5`.
    pub fn set_tolerance(mut self, eps: F, t: F) -> Self {
        self.eps = eps;
        self.t = t;
        self
    }
}

impl<O, F> Solver<O> for Brent<F>
where
    O: ArgminOp<Param = F, Output = F, Float = F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Brent";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        // Brent maintains its own state
        _state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let u = self.a + self.c * (self.b - self.a);
        self.v = u;
        self.w = u;
        self.x = u;
        let f = op.apply(&u)?;
        self.fv = f;
        self.fw = f;
        self.fx = f;
        Ok(Some(ArgminIterData::new().param(self.x).cost(self.fx)))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        // Brent maintains its own state
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let two = F::from_f64(2f64).unwrap();
        let tol = self.eps * self.x.abs() + self.t;
        let m = (self.a + self.b) / two;
        if (self.x - m).abs() <= two * tol - (self.b - self.a) / two {
            return Ok(ArgminIterData::new()
                .termination_reason(TerminationReason::TargetPrecisionReached)
                .param(self.x)
                .cost(self.fx));
        }
        let p = (self.x - self.v) * (self.x - self.v) * (self.fx - self.fw)
            - (self.x - self.w) * (self.x - self.w) * (self.fx - self.fv);
        let q = two
            * ((self.x - self.w) * (self.fx - self.fv) - (self.x - self.v) * (self.fx - self.fw));
        let (p, q) = if q >= F::zero() { (p, q) } else { (-p, -q) };
        self.d = if self.e.abs() <= tol
            || p < q * (self.a - self.x)
            || p > q * (self.b - self.x)
            || two * p.abs() >= q * self.e.abs()
        {
            // golden section step
            self.e = if self.x < m { self.b } else { self.a } - self.x;
            self.c * self.e
        } else {
            // parabolic interpolation step
            self.e = self.d;
            let d = p / q;
            // f must not be evaluated too close from a and b
            if self.x + d - self.a < two * tol || self.b - self.x - d < two * tol {
                (m - self.x).signum() * tol
            } else {
                d
            }
        };
        // f must not be evaluated too close from x
        self.u = self.x
            + if self.d.abs() >= tol {
                self.d
            } else {
                self.d.signum() * tol
            };
        let fu = op.apply(&self.u)?;
        if fu <= self.fx {
            if self.u < self.x {
                self.b = self.x;
            } else {
                self.a = self.x;
            }
            // v is the previous w
            self.v = self.w;
            self.fv = self.fw;
            // w is the second lowest value (former x)
            self.w = self.x;
            self.fw = self.fx;
            // x is the lowest value (u)
            self.x = self.u;
            self.fx = fu;
        } else {
            if self.u < self.x {
                self.a = self.u;
            } else {
                self.b = self.u;
            }
            if fu <= self.fw || self.w == self.x {
                self.v = self.w;
                self.fv = self.fw;
                self.w = self.u;
                self.fw = fu;
            } else if fu <= self.fv || self.v == self.x || self.v == self.w {
                self.v = self.u;
                self.fv = fu;
            }
        }
        Ok(ArgminIterData::new().param(self.x).cost(self.fx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(brent, Brent<f64>);

    struct TestFunc {}
    impl ArgminOp for TestFunc {
        type Param = f64;
        type Output = f64;
        type Float = f64;
        type Jacobian = ();
        type Hessian = ();

        fn apply(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            Ok((-x).exp() - (5. - x / 2.).exp())
        }
    }
    #[test]
    fn test_brent() {
        let cost = TestFunc {};
        let solver = Brent::new(-10., 10.);
        let res = Executor::new(cost, solver, f64::nan())
            .max_iters(13)
            .run()
            .unwrap();
        assert_eq!(
            res.state().termination_reason,
            TerminationReason::TargetPrecisionReached
        );
        assert_eq!(res.state().param, -8.613701289624956);
        assert_eq!(res.state().prev_param, -8.613701289624956);
        assert_eq!(res.state().best_param, -8.613701289624956);
        assert_eq!(res.state().prev_best_param, -8.613701289624956);
        assert_eq!(res.state().cost, -5506.616448675639);
        assert_eq!(res.state().best_cost, -5506.616448675639);
        assert_eq!(res.state().prev_cost, -5506.616448675639);
        assert_eq!(res.state().prev_best_cost, -5506.616448675639);
        assert_eq!(res.state().iter, 13);
        assert_eq!(res.state().cost_func_count, 13);
    }
}
