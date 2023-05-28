extern crate nalgebra as na;

use anyhow::Result;
use na::{Quaternion, Rotation3, SMatrix, SVector, Unit, UnitQuaternion, Vector3};
use tracing::{event, Level};
use tracing_subscriber;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let t_orbit = 5926.0;
    let n_orbit = 3.0;
    let t_start = 0.0;
    let t_end = t_orbit * n_orbit;
    let dt_0 = 0.1;
    let dt_control = 10.0;
    let r_0 = UnitQuaternion::from_euler_angles(0.0, -15.0, 0.0)
        .transform_vector(&SVector::<f64, 3>::new(7078.0e3, 0.0, 0.0)); // Initial orbital position
    let v_0 = UnitQuaternion::from_euler_angles(0.0, -15.0, 0.0)
        .transform_vector(&SVector::<f64, 3>::new(0.0, 7.5043e3, 0.0)); // Initial orbital velocity
    let q_0 = UnitQuaternion::from_euler_angles(10.0, 10.0, 10.0); // Initial satellite attitude
    let w_0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0); // Initial body angular velocity

    let x_0 = SVector::from([
        r_0.index(0).to_owned(),
        r_0.index(1).to_owned(),
        r_0.index(2).to_owned(),
        v_0.index(0).to_owned(),
        v_0.index(1).to_owned(),
        v_0.index(2).to_owned(),
        q_0.i,
        q_0.j,
        q_0.k,
        q_0.w,
        w_0.index(0).to_owned(),
        w_0.index(1).to_owned(),
        w_0.index(2).to_owned(),
    ]);
    let mut t = t_start;
    let mut t_prev_control_step = t_start;

    let solver = RK45 {
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
        "t", "x", "y", "z", "v1", "v2", "v3", "q1", "q2", "q3", "q0", "w1", "w2", "w3", "q1_lvlh",
        "q2_lvlh", "q3_lvlh", "q0_lvlh", "w1_lvlh", "w2_lvlh", "w3_lvlh", "obv", "u1", "u2", "u3",
    ])?;
    let mut x_obv = observer.step(SVector::<f64, 1>::new(0.0));
    let mut u = controller.step(x_obv);
    let mut x = x_0;
    let mut dt = dt_0;
    wtr.serialize((t, x, lvlh(x), x_obv, u))?;

    // Loop
    while t < t_end {
        let (x_new, dt_new) = solver.step(x, u, combined, t, dt)?;

        let t_next_control_step = t_prev_control_step + dt_control;

        if (t + dt) > t_next_control_step {
            (x, dt) = solver.step_to(x, u, combined, t, t_next_control_step);
            t = t + dt;
            x_obv = observer.step(SVector::<f64, 1>::new(0.0));
            u = controller.step(x_obv);
            t_prev_control_step = t;
        } else {
            t = t + dt;
            // A step_to with dt = 0.0 blows up the next interation
            // Cannot use dt from forced step
            //(x, dt) = (x_new, dt_new);
            x = x_new;
        }
        dt = dt_new;

        wtr.serialize((t, x, lvlh(x), x_obv, u))?;
    }

    wtr.flush()?;
    Ok(())
}

fn combined(x: SVector<f64, 13>, u: SVector<f64, 3>) -> SVector<f64, 13> {
    let r = SVector::<f64, 3>::from([
        x.index(0).to_owned(),
        x.index(1).to_owned(),
        x.index(2).to_owned(),
    ]);
    let v = SVector::<f64, 3>::from([
        x.index(3).to_owned(),
        x.index(4).to_owned(),
        x.index(5).to_owned(),
    ]);
    let q = Quaternion::from([
        x.index(6).to_owned(),
        x.index(7).to_owned(),
        x.index(8).to_owned(),
        x.index(9).to_owned(),
    ]);
    let w = SVector::<f64, 3>::from([
        x.index(10).to_owned(),
        x.index(11).to_owned(),
        x.index(12).to_owned(),
    ]);

    let r_dot = orbit_kinematics(r, v);
    let v_dot = orbit_dynamics(r, v);
    let q_dot = attitude_kinematics(q, w);
    let w_dot = attitude_dynamics(q, w, r, u);

    let x_dot = SVector::<f64, 13>::from([
        r_dot.index(0).to_owned(),
        r_dot.index(1).to_owned(),
        r_dot.index(2).to_owned(),
        v_dot.index(0).to_owned(),
        v_dot.index(1).to_owned(),
        v_dot.index(2).to_owned(),
        q_dot.i,
        q_dot.j,
        q_dot.k,
        q_dot.w,
        w_dot.index(0).to_owned(),
        w_dot.index(1).to_owned(),
        w_dot.index(2).to_owned(),
    ]);

    return x_dot;
}

/// Orbit kinematics
///
/// # Arguments
///
/// * r - Orbital position (inertial)
/// * v - Orbital velocity (inertial)
fn orbit_kinematics(r: SVector<f64, 3>, v: SVector<f64, 3>) -> SVector<f64, 3> {
    let w = r.cross(&v) / r.norm_squared();

    return w.cross(&r);
}

/// Orbit dynamics
///
/// # Arguments
///
/// * r - Orbital postion (intertial)
/// * v - Orbital velocity (inertial)
fn orbit_dynamics(r: SVector<f64, 3>, v: SVector<f64, 3>) -> SVector<f64, 3> {
    let w = r.cross(&v) / r.norm_squared();

    return -w.norm_squared() * r;
}

/// Satellite kinematics
///
/// # Arguments
///
/// * `q` - Quaternion attitude (body relative to inertial)
/// * `w` - Anuglar velocity (body relative to inertial, expressed in body)
fn attitude_kinematics(q: Quaternion<f64>, w: SVector<f64, 3>) -> Quaternion<f64> {
    return 0.5 * Quaternion::new(0.0, w[0], w[1], w[2]) * q;
}

/// Satellite dynamics
///
/// # Arguments
///
/// * `q` - Quaternion attitude (body relative to inertial)
/// * `w` - Angular velocity (body relative to inertial, expressed in body)
/// * `r` - Orbital position (inertial)
/// * `u` - Control torques (expressed in body)
fn attitude_dynamics(
    q: Quaternion<f64>,
    w: SVector<f64, 3>,
    r: SVector<f64, 3>,
    u: SVector<f64, 3>,
) -> SVector<f64, 3> {
    let inertia =
        SMatrix::<f64, 3, 3>::from_diagonal(&SVector::<f64, 3>::new(2500.0, 2300.0, 3100.0));

    //return inertia.try_inverse().unwrap()
    //    * ((-w).cross(&(inertia * w)) + torque_gg(r, inertia) + torque_d());

    return inertia.try_inverse().unwrap() * ((-w).cross(&(inertia * w)));
}

/// Gravity gradient torque
///
/// # Arguments
///
/// * `r` - Orbital position (inertial)
/// * `inertia` - Spacecraft inerital matrix
fn torque_gg(r: SVector<f64, 3>, inertia: SMatrix<f64, 3, 3>) -> SVector<f64, 3> {
    let mu = 3.986004418E14; // Earth gravitational parameter

    return (3.0 * mu / f64::powf(r.magnitude(), 3.0))
        * (-r.normalize()).cross(&(inertia * -r.normalize()));
}

/// Disturbance torque
fn torque_d() -> SVector<f64, 3> {
    return SVector::<f64, 3>::new(0.001, 0.001, 0.001);
}

fn lvlh(x: SVector<f64, 13>) -> (Quaternion<f64>, SVector<f64, 3>) {
    let r = SVector::<f64, 3>::from([
        x.index(0).to_owned(),
        x.index(1).to_owned(),
        x.index(2).to_owned(),
    ]);

    let v = SVector::<f64, 3>::from([
        x.index(3).to_owned(),
        x.index(4).to_owned(),
        x.index(5).to_owned(),
    ]);

    let q = Quaternion::from([
        x.index(6).to_owned(),
        x.index(7).to_owned(),
        x.index(8).to_owned(),
        x.index(9).to_owned(),
    ]);

    let w = SVector::<f64, 3>::from([
        x.index(10).to_owned(),
        x.index(11).to_owned(),
        x.index(12).to_owned(),
    ]);

    let w_o = SVector::<f64, 3>::new(0.0, v.norm() / r.norm(), 0.0);

    let r = -r.normalize();
    let v = v.normalize();

    //let rot = Rotation3::from_basis_unchecked(&[v, n, r]);

    //let angle = rot.angle() / 2.0;
    //let axis = match rot.axis() {
    //    Some(axis) => axis,
    //    None => Vector3::x_axis(),
    //};

    //let q_lvlh = Quaternion::new(
    //    angle.cos(),
    //    axis[0] * angle.sin(),
    //    axis[1] * angle.sin(),
    //    axis[2] * angle.sin(),
    //);

    //let w_lvlh = q_lvlh * Quaternion::new(0.0, w[0], w[1], w[2]) * q_lvlh.conjugate();
    //let w_lvlh = SVector::<f64, 3>::new(w_lvlh.i, w_lvlh.j, w_lvlh.k);

    let z_eci = SVector::<f64, 3>::new(0.0, 0.0, 1.0);
    let a1 = z_eci.cross(&r);
    let q1 = Quaternion::new(1.0 + z_eci.dot(&r), a1[0], a1[1], a1[2]).normalize();

    let x_eci = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let x_q1 = q1 * x_eci * q1.conjugate();
    let x_q1 = SVector::<f64, 3>::new(x_q1.i, x_q1.j, x_q1.k);

    // Dot product provides no information if vector A is clockwise or counter-clockwise from vector B
    let angle: f64;
    if (x_q1.cross(&v) * r.transpose())[0].asin() > 0.0 {
        angle = (x_q1.dot(&v)).acos() / 2.0;
    } else {
        angle = -(x_q1.dot(&v)).acos() / 2.0;
    }
    let axis = r;
    let q2 = Quaternion::new(
        angle.cos(),
        axis[0] * angle.sin(),
        axis[1] * angle.sin(),
        axis[2] * angle.sin(),
    );

    let q_lvlh = q2 * q1;
    let w_lvlh = q_lvlh * Quaternion::new(0.0, w[0], w[1], w[2]) * q_lvlh.conjugate();
    let w_lvlh = SVector::<f64, 3>::new(w_lvlh.i, w_lvlh.j, w_lvlh.k) + w_o;

    return (q_lvlh, w_lvlh);
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
//const A1: f64 = 0.0;
//const A2: f64 = 2.0 / 9.0;
//const A3: f64 = 1.0 / 3.0;
//const A4: f64 = 3.0 / 4.0;
//const A5: f64 = 1.0;
//const A6: f64 = 5.0 / 6.0;
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
        let dt = t_target - t;

        let increments = self.compute_increments(x, u, df, t, dt);
        let solution = self.compute_solution(x, &increments);
        let truncation_error = self.compute_truncation_error(&increments);

        (solution, dt)
    }
}
