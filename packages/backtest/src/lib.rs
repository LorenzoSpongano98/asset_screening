//! PyO3 bindings for the backtest engine.

pub mod costs;
pub mod engine;
pub mod portfolio;

#[pyo3::pymodule]
mod backtest_engine {
    use pyo3::prelude::*;

    #[pyfunction]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}
