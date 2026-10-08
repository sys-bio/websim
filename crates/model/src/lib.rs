//! The model layer of websim, with no user-interface code: reading models
//! written in a subset of Antimony, conservation analysis, steady states, and
//! simulation. Kept separate from the app so the bifurcation library
//! (`bifurcata-rs`) can build on it.

pub mod antimony;
pub mod conservation;
pub mod linalg;
pub mod model;
pub mod newton;
pub mod ode;
pub mod solvers;
pub mod steady;
