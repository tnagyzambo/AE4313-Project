extern crate nalgebra as na;

use anyhow::Result;
use na::{Quaternion, SMatrix, SVector};
use tracing::{event, Level};
use tracing_subscriber;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let t_start = 0.0;
    let t_end = 100.0;
    let dt_0 = 0.1;
    let dt_control = 10.0;
    let x_0 = SVector::from([0.0, 0.0, 0.0, 1.0, 0.1, 0.1, 0.0]);
    let mut t = t_start;
    let mut t_prev_control_step = t_start;

    let mut solver = RK45 {
        dt_min: 1E-10,
        dt_max: dt_control,
        tol: 1E-10,
    };

    let mut observer = Observer {};

    let mut controller = Controller {
        u: SVector::from([0.0, 0.0, 0.0]),
    };

    let mut wtr = csv::Writer::from_path("out.csv")?;
    wtr.write_record(&[
        "time (s)", "q1", "q2", "q3", "q0", "w1", "w2", "w3", "q1_hat", "q2_hat", "q3_hat",
        "q4_hat", "w1_hat", "w2_hat", "w3_hat", "u1", "u2", "u3",
    ])?;

    // First step with initial conditions
    let mut x_obv = observer.step(x_0);
    let mut u = controller.step(x_obv);
    let (mut x, mut dt) = solver.step(x_0, u, combined, t, dt_0)?;
    t = t + dt;
    wtr.serialize((t, x, x_obv, u))?;

    // Loop
    while t < t_end {
        let (x_new, dt_new) = solver.step(x, u, combined, t, dt)?;

        let t_next_control_step = t_prev_control_step + dt_control;

        if (t + dt) > t_next_control_step {
            (x, dt) = solver.step_to(x, u, combined, t, t_next_control_step);
            t = t + dt;
            x_obv = observer.step(x);
            u = controller.step(x_obv);
            t_prev_control_step = t;
        } else {
            t = t + dt;
            (x, dt) = (x_new, dt_new);
        }

        wtr.serialize((t, x, x_obv, u))?;
    }

    wtr.flush()?;
    Ok(())
}

fn combined(x: SVector<f64, 7>, u: SVector<f64, 3>) -> SVector<f64, 7> {
    let q = Quaternion::from([
        x.index(0).to_owned(),
        x.index(1).to_owned(),
        x.index(2).to_owned(),
        x.index(3).to_owned(),
    ]);
    let omega = SVector::<f64, 3>::from([
        x.index(4).to_owned(),
        x.index(5).to_owned(),
        x.index(6).to_owned(),
    ]);

    let q_dot = kinematics(q, omega);
    let omega_dot = dynamics(omega, u);

    let x_dot = SVector::<f64, 7>::from([
        q_dot.i,
        q_dot.j,
        q_dot.k,
        q_dot.w,
        omega_dot.index(0).to_owned(),
        omega_dot.index(1).to_owned(),
        omega_dot.index(2).to_owned(),
    ]);

    return x_dot;
}

/// Satellite kinematics
///
/// # Arguments
///
/// * `q` - Quaternion attitude (body relative to inertial)
/// * `omega` - Anuglar velocity (body relative to inertial, expressed in body)
fn kinematics(q: Quaternion<f64>, omega: SVector<f64, 3>) -> Quaternion<f64> {
    let phi = SMatrix::<f64, 4, 4>::new(
        0.0, omega[2], -omega[1], omega[0], //
        -omega[2], 0.0, omega[0], omega[1], //
        omega[1], -omega[0], 0.0, omega[2], //
        -omega[0], -omega[1], -omega[2], 0.0,
    );

    return Quaternion::from_vector(0.5 * phi * q.as_vector());
}

/// Satellite dynamics
///
/// # Arguments
///
/// * `omega` - Angular velocity (body relative to inertial, expressed in body)
/// * `u` - Control torques (expressed in body)
fn dynamics(omega: SVector<f64, 3>, u: SVector<f64, 3>) -> SVector<f64, 3> {
    let inertia =
        SMatrix::<f64, 3, 3>::from_diagonal(&SVector::<f64, 3>::new(2500.0, 2300.0, 3100.0));

    return inertia.try_inverse().unwrap() * (-omega.cross(&(inertia * omega)));
}

struct Observer<const N: usize> {}

impl<const N: usize> Observer<N> {
    fn step(&self, x: SVector<f64, N>) -> SVector<f64, N> {
        return x;
    }
}

struct Controller<const N: usize> {
    u: SVector<f64, 3>,
}

impl<const N: usize> Controller<N> {
    fn step(&mut self, x: SVector<f64, N>) -> SVector<f64, 3> {
        self.u = self.u + SVector::from([1.0, 0.0, 0.0]);
        self.u
    }
}

// Magic numbers for RK45 solver
const A1: f64 = 0.0;
const A2: f64 = 2.0 / 9.0;
const A3: f64 = 1.0 / 3.0;
const A4: f64 = 3.0 / 4.0;
const A5: f64 = 1.0;
const A6: f64 = 5.0 / 6.0;
const B21: f64 = 2.0 / 9.0;
const B31: f64 = 1.0 / 12.0;
const B32: f64 = 1.0 / 4.0;
const B41: f64 = 69.0 / 128.0;
const B42: f64 = -243.0 / 128.0;
const B43: f64 = 135.0 / 64.0;
const B51: f64 = -17.0 / 12.0;
const B52: f64 = 27.0 / 4.0;
const B53: f64 = -27.0 / 5.0;
const B54: f64 = 16.0 / 15.0;
const B61: f64 = 65.0 / 432.0;
const B62: f64 = -5.0 / 16.0;
const B63: f64 = 13.0 / 16.0;
const B64: f64 = 4.0 / 27.0;
const B65: f64 = 5.0 / 144.0;
const CH1: f64 = 47.0 / 450.0;
const CH2: f64 = 0.0;
const CH3: f64 = 12.0 / 25.0;
const CH4: f64 = 32.0 / 225.0;
const CH5: f64 = 1.0 / 30.0;
const CH6: f64 = 6.0 / 25.0;
const CT1: f64 = 1.0 / 150.0;
const CT2: f64 = 0.0;
const CT3: f64 = -3.0 / 100.0;
const CT4: f64 = 16.0 / 75.0;
const CT5: f64 = 1.0 / 20.0;
const CT6: f64 = -6.0 / 25.0;

#[derive(Debug)]
enum SolverError {
    DTMinExceeded,
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::DTMinExceeded => {
                write!(f, "dt_min exceed, unable to solve with required tolerance")
            }
        }
    }
}

impl std::error::Error for SolverError {}

struct RK45Increments<const N: usize> {
    k1: SVector<f64, N>,
    k2: SVector<f64, N>,
    k3: SVector<f64, N>,
    k4: SVector<f64, N>,
    k5: SVector<f64, N>,
    k6: SVector<f64, N>,
}

/// Reference found [here](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_methoxd).
struct RK45<const N: usize, const M: usize> {
    dt_min: f64,
    dt_max: f64,
    tol: f64,
}

impl<const N: usize, const M: usize> RK45<N, M> {
    fn compute_increments(
        &self,
        x: SVector<f64, N>,
        u: SVector<f64, M>,
        df: fn(SVector<f64, N>, SVector<f64, M>) -> SVector<f64, N>,
        t: f64,
        dt: f64,
    ) -> RK45Increments<N> {
        let k1 = dt * (df)(x, u);
        let k2 = dt * (df)(x + B21 * k1, u);
        let k3 = dt * (df)(x + B31 * k1 + B32 * k2, u);
        let k4 = dt * (df)(x + B41 * k1 + B42 * k2 + B43 * k3, u);
        let k5 = dt * (df)(x + B51 * k1 + B52 * k2 + B53 * k3 + B54 * k4, u);
        let k6 = dt * (df)(x + B61 * k1 + B62 * k2 + B63 * k3 + B64 * k4 + B65 * k5, u);

        RK45Increments {
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
        }
    }

    fn compute_solution(
        &self,
        x: SVector<f64, N>,
        increments: &RK45Increments<N>,
    ) -> SVector<f64, N> {
        x + CH1 * increments.k1
            + CH2 * increments.k2
            + CH3 * increments.k3
            + CH4 * increments.k4
            + CH5 * increments.k5
            + CH6 * increments.k6
    }

    fn compute_truncation_error(&self, increments: &RK45Increments<N>) -> f64 {
        (CT1 * increments.k1
            + CT2 * increments.k2
            + CT3 * increments.k3
            + CT4 * increments.k4
            + CT5 * increments.k5
            + CT6 * increments.k6)
            .norm()
    }

    fn compute_dt(&self, dt: f64, truncation_error: f64) -> Result<f64, SolverError> {
        let mut dt_new = 0.9 * dt * f64::powf(self.tol / truncation_error, 1.0 / 5.0);

        if dt_new < self.dt_min {
            event!(
                Level::ERROR,
                "computed time step: {}s is smaller than the minimum: {}s",
                dt_new,
                self.dt_min
            );
            return Err(SolverError::DTMinExceeded);
        } else if dt_new > self.dt_max {
            event!(
                Level::WARN,
                "computed time step: {}s is larger than the maximum: {}s",
                dt_new,
                self.dt_max
            );
            dt_new = self.dt_max;
        }

        Ok(dt_new)
    }

    fn step(
        &self,
        x: SVector<f64, N>,
        u: SVector<f64, M>,
        df: fn(SVector<f64, N>, SVector<f64, M>) -> SVector<f64, N>,
        t: f64,
        dt: f64,
    ) -> Result<(SVector<f64, N>, f64), SolverError> {
        event!(Level::INFO, "step");
        let increments = self.compute_increments(x, u, df, t, dt);
        let solution = self.compute_solution(x, &increments);
        let truncation_error = self.compute_truncation_error(&increments);
        let dt_new = self.compute_dt(dt, truncation_error)?;

        // If truncation error exceeds the tolerance rerun step with newly computed timestep
        if truncation_error > self.tol {
            self.step(x, u, df, t, dt_new)?;
        }

        Ok((solution, dt_new))
    }

    fn step_to(
        &self,
        x: SVector<f64, N>,
        u: SVector<f64, M>,
        df: fn(SVector<f64, N>, SVector<f64, M>) -> SVector<f64, N>,
        t: f64,
        t_target: f64,
    ) -> (SVector<f64, N>, f64) {
        event!(Level::INFO, "step_to");
        let dt = t_target - t;

        let increments = self.compute_increments(x, u, df, t, dt);
        let solution = self.compute_solution(x, &increments);
        let truncation_error = self.compute_truncation_error(&increments);

        (solution, dt)
    }
}
