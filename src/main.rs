extern crate nalgebra as na;

use na::{SMatrix, SVector};
use std::{error::Error, io, process};

fn main() -> Result<(), Box<dyn Error>> {
    let mut solver = RK45 {
        df: deriv,
        x: SVector::<f64, 2>::new(1.0, 0.0),
        x0: SVector::<f64, 2>::new(1.0, 1.0),
        dt: 0.01,
        dt_min: 0.0,
        dt_max: 0.0,
        t: 0.0,
        t_start: 0.0,
        t_end: 100.0,
        tol: 0.0000001,
    };

    let mut wtr = csv::Writer::from_writer(io::stdout());
    for i in 1..100 {
        wtr.serialize((solver.t, solver.x))?;
        solver.step();
    }

    wtr.flush()?;
    Ok(())
}

fn deriv(_t: f64, x: SVector<f64, 2>) -> SVector<f64, 2> {
    let A = SMatrix::<f64, 2, 2>::new(-0.5572, -0.7814, 0.7814, 0.0);
    return A * x;
}

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

// https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
struct RK45<const N: usize> {
    df: fn(f64, SVector<f64, N>) -> SVector<f64, N>,
    x: SVector<f64, N>,
    x0: SVector<f64, N>,
    dt: f64,
    dt_min: f64,
    dt_max: f64,
    t: f64,
    t_start: f64,
    t_end: f64,
    tol: f64,
}

impl<const N: usize> RK45<N> {
    fn step(&mut self) {
        let k1 = self.dt * (self.df)(self.t + A1, self.x);
        let k2 = self.dt * (self.df)(self.t + A2 * self.dt, self.x + B21 * k1);
        let k3 = self.dt * (self.df)(self.t + A3 * self.dt, self.x + B31 * k1 + B32 * k2);
        let k4 = self.dt
            * (self.df)(
                self.t + A4 * self.dt,
                self.x + B41 * k1 + B42 * k2 + B43 * k3,
            );
        let k5 = self.dt
            * (self.df)(
                self.t + A5 * self.dt,
                self.x + B51 * k1 + B52 * k2 + B53 * k3 + B54 * k4,
            );
        let k6 = self.dt
            * (self.df)(
                self.t + A6 * self.dt,
                self.x + B61 * k1 + B62 * k2 + B63 * k3 + B64 * k4 + B65 * k5,
            );

        self.x = self.x + CH1 * k1 + CH2 * k2 + CH3 * k3 + CH4 * k4 + CH5 * k5 + CH6 * k6;

        let truncation_error =
            (CT1 * k1 + CT2 * k2 + CT3 * k3 + CT4 * k4 + CT5 * k5 + CT6 * k6).norm();

        self.dt = 0.9 * self.dt * f64::powf(self.tol / truncation_error, 1.0 / 5.0);

        if truncation_error > self.tol {
            self.step();
        } else {
            self.t = self.t + self.dt;
        }
    }
}
