//! IR optimization passes.

use std::collections::HashMap;

use crate::eval::{eval_binop, eval_unaryop};
use crate::ir::{Ir, IrProgram, Reg, Value};

/// Constant folding: evaluate operations on known constants at compile time.
pub fn constant_folding(program: &mut IrProgram) {
    let mut known: HashMap<Reg, Value> = HashMap::new();

    for ir in &mut program.instructions {
        match ir {
            Ir::LoadConst { dst, val } => {
                known.insert(*dst, val.clone());
            }
            Ir::BinOp { op, dst, lhs, rhs } => {
                if let (Some(lv), Some(rv)) = (known.get(lhs), known.get(rhs)) {
                    if let Some(result) = eval_binop(*op, lv, rv) {
                        known.insert(*dst, result.clone());
                        *ir = Ir::LoadConst {
                            dst: *dst,
                            val: result,
                        };
                    }
                }
            }
            Ir::UnaryOp { op, dst, src } => {
                if let Some(sv) = known.get(src) {
                    if let Some(result) = eval_unaryop(*op, sv) {
                        known.insert(*dst, result.clone());
                        *ir = Ir::LoadConst {
                            dst: *dst,
                            val: result,
                        };
                    }
                }
            }
            Ir::Copy { dst, src } => {
                if let Some(v) = known.get(src).cloned() {
                    known.insert(*dst, v);
                }
            }
            _ => {}
        }
    }
}

/// Constant propagation: replace register reads with known constants.
pub fn constant_propagation(program: &mut IrProgram) {
    let mut known: HashMap<Reg, Value> = HashMap::new();

    for ir in &mut program.instructions {
        match ir {
            Ir::LoadConst { dst, val } => {
                known.insert(*dst, val.clone());
            }
            Ir::Copy { dst, src } => {
                if let Some(v) = known.get(src).cloned() {
                    known.insert(*dst, v.clone());
                    *ir = Ir::LoadConst { dst: *dst, val: v };
                }
            }
            _ => {}
        }
    }
}

/// Dead code elimination: remove nop instructions and unreachable code after unconditional jumps.
pub fn dead_code_elimination(program: &mut IrProgram) {
    let mut i = 0;
    while i < program.instructions.len() {
        if matches!(program.instructions[i], Ir::Jump(_)) {
            let mut j = i + 1;
            while j < program.instructions.len() {
                if matches!(program.instructions[j], Ir::Label(_)) {
                    break;
                }
                program.instructions[j] = Ir::Nop;
                j += 1;
            }
        }
        i += 1;
    }

    program.instructions.retain(|ir| !matches!(ir, Ir::Nop));
}

/// Peephole optimization: simplify trivial patterns.
pub fn peephole(program: &mut IrProgram) {
    for ir in &mut program.instructions {
        if matches!(ir, Ir::Copy { dst, src } if *dst == *src) {
            *ir = Ir::Nop;
        }
    }

    program.instructions.retain(|ir| !matches!(ir, Ir::Nop));
}

/// Run all optimization passes.
pub fn optimize(program: &mut IrProgram) {
    constant_folding(program);
    constant_propagation(program);
    dead_code_elimination(program);
    peephole(program);
}
