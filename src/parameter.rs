/// TODO DOCUMENTATION
///
use errors::*;
use std::fmt::Debug;
use std::default::Default;
// use num::ToPrimitive;
use rand;
use rand::distributions::{IndependentSample, Range};
use ndarray::Array1;
use ArgminCostValue;

/// This trait needs to be implemented for every parameter fed into the solvers.
/// This is highly *UNSTABLE* and will change in the future.
pub trait ArgminParameter: Clone + Default + Debug {
    /// Defines a modification of the parameter vector.
    ///
    /// The parameters:
    ///
    /// `&self`: reference to the object of type `Self`
    /// `lower_bound`: Lower bound of the parameter vector. Same type as parameter vector (`Self`)
    /// `upper_bound`: Upper bound of the parameter vector. Same type as parameter vector (`Self`)
    /// `constraint`: Additional (non)linear constraint whith the signature `&Fn(&Self) -> bool`.
    /// The provided function takes a parameter as input and returns `true` if the parameter vector
    /// satisfies the constraints and `false` otherwise.
    fn modify(&self, &Self, &Self, &Fn(&Self) -> bool) -> Self;

    /// Returns a completely random parameter vector
    ///
    /// The resulting parameter vector satisfies `lower_bound`, `upper_bound`.
    fn random(&Self, &Self) -> Result<Self>;
}

impl ArgminParameter for Vec<f64> {
    fn modify(
        &self,
        lower_bound: &Vec<f64>,
        upper_bound: &Vec<f64>,
        constraint: &Fn(&Vec<f64>) -> bool,
    ) -> Vec<f64> {
        let step = Range::new(0, self.len());
        let range = Range::new(-1.0_f64, 1.0_f64);
        let mut rng = rand::thread_rng();
        let mut param = self.clone();
        loop {
            let idx = step.ind_sample(&mut rng);
            param[idx] = self[idx] + range.ind_sample(&mut rng);
            if param[idx] < lower_bound[idx] {
                param[idx] = lower_bound[idx];
            }
            if param[idx] > upper_bound[idx] {
                param[idx] = upper_bound[idx];
            }
            if constraint(&param) {
                break;
            }
        }
        param
    }

    fn random(lower_bound: &Vec<f64>, upper_bound: &Vec<f64>) -> Result<Vec<f64>> {
        let mut out = vec![];
        let mut rng = rand::thread_rng();
        for elem in lower_bound.iter().zip(upper_bound.iter()) {
            if elem.0 >= elem.1 {
                return Err(ErrorKind::InvalidParameter(
                    "Parameter: lower_bound must be lower than upper_bound.".into(),
                ).into());
            }
            let range = Range::new(*elem.0, *elem.1);
            out.push(range.ind_sample(&mut rng));
        }
        Ok(out)
    }
}

impl<T> ArgminParameter for Array1<T>
where
    T: ArgminCostValue + rand::distributions::range::SampleRange,
{
    fn modify(
        &self,
        lower_bound: &Array1<T>,
        upper_bound: &Array1<T>,
        constraint: &Fn(&Array1<T>) -> bool,
    ) -> Array1<T> {
        let step = Range::new(0, self.len());
        // let range = Range::new(-1.0_f64, 1.0_f64);
        let range = Range::new(
            T::from_f64(-1.0_f64).unwrap(),
            T::from_f64(1.0_f64).unwrap(),
        );
        let mut rng = rand::thread_rng();
        let mut param = self.clone();
        loop {
            let idx = step.ind_sample(&mut rng);
            param[idx] = self[idx] + range.ind_sample(&mut rng);
            if param[idx] < lower_bound[idx] {
                param[idx] = lower_bound[idx];
            }
            if param[idx] > upper_bound[idx] {
                param[idx] = upper_bound[idx];
            }
            if constraint(&param) {
                break;
            }
        }
        param
    }

    fn random(lower_bound: &Array1<T>, upper_bound: &Array1<T>) -> Result<Array1<T>> {
        // unimplemented!()
        let mut rng = rand::thread_rng();
        let out: Array1<T> = lower_bound
            .iter()
            .zip(upper_bound.iter())
            .map(|(a, b)| {
                if a >= b {
                    panic!("Parameter: lower_bound must be lower than upper_bound.");
                    // Unfortunately the following doesnt work here!
                    // return Err(ErrorKind::InvalidParameter(
                    //     "Parameter: lower_bound must be lower than upper_bound.".into(),
                    // ).into());
                }
                let range = Range::new(*a, *b);
                range.ind_sample(&mut rng)
            })
            .collect();
        Ok(out)
    }
}
