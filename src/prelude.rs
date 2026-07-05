//! Convenience re-export (= `use alice_compiler::prelude::*;`).

pub use crate::ast::{BinOp, Expr, UnaryOp};
pub use crate::codegen::Codegen;
pub use crate::compile::{compile_and_run, compile_optimize_and_run};
pub use crate::ir::{Ir, IrProgram, LabelId, Reg, Value};
pub use crate::optimization::{
    constant_folding, constant_propagation, dead_code_elimination, optimize, peephole,
};
pub use crate::vm::Vm;
