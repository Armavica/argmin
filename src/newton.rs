/// Newton method
///
/// TODO
use std;
use rand;
use ndarray::*;
// use ndarray::{Array1, Array2, LinalgScalar, ScalarOperand};
use ndarray::linalg::Dot;
use ndarray_linalg::*;
use errors::*;
use problem::Problem;
use result::ArgminResult;
use ArgminSolver;
use ArgminCostValue;

pub trait ArgminNewtonValue
    : ArgminCostValue + rand::distributions::range::SampleRange + ScalarOperand + LinalgScalar
    {
}
impl<T> ArgminNewtonValue for T
where
    T: ArgminCostValue + rand::distributions::range::SampleRange + ScalarOperand + LinalgScalar,
    // Array2<T>: Inverse + Dot<Array1<T>>,
    // Array1<T>: Inverse + Dot<Array2<T>>,
{
}

/// Newton method struct (duh)
pub struct Newton<'a, T>
where
    T: ArgminNewtonValue + 'a,
{
    /// step size
    gamma: f64,
    /// Maximum number of iterations
    max_iters: u64,
    /// current state
    state: NewtonState<'a, T>,
}

/// Indicates the current state of the Newton method
struct NewtonState<'a, T>
where
    T: ArgminNewtonValue + 'a,
{
    /// Reference to the problem. This is an Option<_> because it is initialized as `None`
    problem: Option<&'a Problem<'a, Array1<T>, T, Array2<T>>>,
    /// Current parameter vector
    param: Array1<T>,
    /// Current number of iteration
    iter: u64,
}

impl<'a, T> NewtonState<'a, T>
where
    T: ArgminNewtonValue + 'a,
{
    /// Constructor for `NewtonState`
    pub fn new() -> Self {
        NewtonState {
            problem: None,
            param: Array1::<T>::default(1),
            iter: 0_u64,
        }
    }
}

impl<'a, T> Newton<'a, T>
where
    T: ArgminNewtonValue + 'a,
{
    /// Return a `Newton` struct
    pub fn new() -> Self {
        Newton {
            gamma: 1.0,
            max_iters: std::u64::MAX,
            state: NewtonState::new(),
        }
    }

    /// Set maximum number of iterations
    pub fn max_iters(&mut self, max_iters: u64) -> &mut Self {
        self.max_iters = max_iters;
        self
    }
}

impl<'a, T> ArgminSolver<'a> for Newton<'a, T>
where
    T: ArgminNewtonValue + 'a,
    Array2<T>: Inverse, // + Dot<Array1<T>>,
{
    type A = Array1<T>;
    type B = T;
    type C = Array2<T>;

    /// Initialize with a given problem and a starting point
    fn init(
        &mut self,
        problem: &'a Problem<'a, Array1<T>, T, Array2<T>>,
        init_param: &Array1<T>,
    ) -> Result<()> {
        self.state = NewtonState {
            problem: Some(problem),
            param: init_param.to_owned(),
            iter: 0_u64,
        };
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Array1<T>, T>> {
        // TODO: Move to next point
        // x_{n+1} = x_n - \gamma [Hf(x_n)]^-1 \nabla f(x_n)
        let g = (self.state.problem.unwrap().gradient.unwrap())(&self.state.param);
        // let h_inv: Array2<_> =
        // let h_inv =
        let h_inv: Result<Array2<_>> =
            (&(self.state.problem.unwrap().hessian.unwrap())(&self.state.param)).inv();
        let tmp: Array1<T> = h_inv.unwrap().dot::<Array1<T>>(&g);
        self.state.param = self.state.param.clone() - T::from_f64(self.gamma).unwrap() * tmp;
        // self.state.param.clone() - T::from_f64(self.gamma).unwrap() * h_inv.dot(&g);
        self.state.iter += 1;
        Ok(ArgminResult::new(
            self.state.param.clone(),
            T::default(),
            self.state.iter,
        ))
    }

    /// Indicates whether any of the stopping criteria are met
    fn terminate(&self) -> bool {
        false
    }

    /// Run Newton method
    fn run(
        &mut self,
        problem: &'a Problem<'a, Array1<T>, T, Array2<T>>,
        init_param: &Array1<T>,
    ) -> Result<ArgminResult<Array1<T>, T>> {
        // initialize
        self.init(problem, init_param)?;

        loop {
            self.next_iter()?;
            if self.terminate() {
                break;
            }
        }
        let fin_cost = (problem.cost_function)(&self.state.param);
        Ok(ArgminResult::new(
            self.state.param.clone(),
            fin_cost,
            self.state.iter,
        ))
    }
}

impl<'a, T> Default for Newton<'a, T>
where
    T: ArgminNewtonValue + 'a,
{
    fn default() -> Self {
        Self::new()
    }
}
