#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

//! ALICE-Compiler: DSL/JIT compiler infrastructure.
//!
//! Provides AST nodes, IR (intermediate representation), code generation,
//! optimization passes, and a register-based VM for execution.

pub mod ast;
pub mod codegen;
pub mod compile;
pub(crate) mod eval;
pub mod ir;
pub mod optimization;
pub mod prelude;
pub mod vm;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::ast::*;
pub use crate::codegen::*;
pub use crate::compile::*;
pub use crate::ir::*;
pub use crate::optimization::*;
pub use crate::vm::*;
