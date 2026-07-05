//! Convenience wrappers: compile + optionally optimize + run.

use crate::ast::Expr;
use crate::codegen::Codegen;
use crate::ir::Value;
use crate::optimization::optimize;
use crate::vm::Vm;

/// Compile an AST expression and execute it, returning the result.
///
/// # Errors
///
/// Returns an error string if compilation or execution fails.
pub fn compile_and_run(expr: &Expr) -> Result<Value, String> {
    let codegen = Codegen::new();
    let program = codegen.compile(expr);
    let mut vm = Vm::new();
    vm.execute(&program)
}

/// Compile and run with optimization passes applied.
///
/// # Errors
///
/// Returns an error string if compilation or execution fails.
pub fn compile_optimize_and_run(expr: &Expr) -> Result<Value, String> {
    let codegen = Codegen::new();
    let mut program = codegen.compile(expr);
    optimize(&mut program);
    let mut vm = Vm::new();
    vm.execute(&program)
}
