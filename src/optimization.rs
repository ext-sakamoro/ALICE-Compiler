//! IR optimization passes.

use std::collections::HashMap;

use crate::eval::{eval_binop, eval_unaryop};
use crate::ir::{Ir, IrProgram, Reg, Value};
use crate::vm::{check_program, VmLimits};

/// Bytes of strings one pass may make. There is no outside spec for this; it is the
/// `max_string_bytes` default of the VM, chosen because a pass holds what a run would.
pub const MAX_PASS_STRING_BYTES: usize = VmLimits::DEFAULT.max_string_bytes;

/// The string bytes left to a pass. A pass that runs out stops where it is: what it has rewritten
/// so far means the same as before, and the rest stays for the VM, whose own limit then applies.
struct Budget(usize);

impl Budget {
    /// Whether `val` can be made `times` more times, and takes them off.
    fn spend(&mut self, val: &Value, times: usize) -> bool {
        let Value::Str(s) = val else { return true };
        match s.len().checked_mul(times).filter(|n| *n <= self.0) {
            Some(n) => {
                self.0 -= n;
                true
            }
            None => false,
        }
    }
}

/// Constant folding: evaluate operations on known constants at compile time.
pub fn constant_folding(program: &mut IrProgram) {
    let mut known: HashMap<Reg, Value> = HashMap::new();
    let mut budget = Budget(MAX_PASS_STRING_BYTES);

    for ir in &mut program.instructions {
        match ir {
            Ir::LoadConst { dst, val } => {
                if !budget.spend(val, 1) {
                    return;
                }
                known.insert(*dst, val.clone());
            }
            Ir::BinOp { op, dst, lhs, rhs } => {
                if let (Some(lv), Some(rv)) = (known.get(lhs), known.get(rhs)) {
                    if let Some(result) = eval_binop(*op, lv, rv) {
                        if !budget.spend(&result, 2) {
                            return;
                        }
                        known.insert(*dst, result.clone());
                        *ir = Ir::LoadConst {
                            dst: *dst,
                            val: result,
                        };
                        continue;
                    }
                }
                known.remove(dst);
            }
            Ir::UnaryOp { op, dst, src } => {
                if let Some(sv) = known.get(src) {
                    if let Some(result) = eval_unaryop(*op, sv) {
                        known.insert(*dst, result.clone());
                        *ir = Ir::LoadConst {
                            dst: *dst,
                            val: result,
                        };
                        continue;
                    }
                }
                known.remove(dst);
            }
            Ir::Copy { dst, src } => {
                if let Some(v) = known.get(src) {
                    if !budget.spend(v, 1) {
                        return;
                    }
                    let v = v.clone();
                    known.insert(*dst, v);
                } else {
                    known.remove(dst);
                }
            }
            // another path can come here with other values
            Ir::Label(_) => known.clear(),
            _ => {}
        }
    }
}

/// Constant propagation: replace register reads with known constants.
pub fn constant_propagation(program: &mut IrProgram) {
    let mut known: HashMap<Reg, Value> = HashMap::new();
    let mut budget = Budget(MAX_PASS_STRING_BYTES);

    for ir in &mut program.instructions {
        match ir {
            Ir::LoadConst { dst, val } => {
                if !budget.spend(val, 1) {
                    return;
                }
                known.insert(*dst, val.clone());
            }
            Ir::Copy { dst, src } => {
                if let Some(v) = known.get(src) {
                    if !budget.spend(v, 2) {
                        return;
                    }
                    let v = v.clone();
                    known.insert(*dst, v.clone());
                    *ir = Ir::LoadConst { dst: *dst, val: v };
                } else {
                    known.remove(dst);
                }
            }
            Ir::BinOp { dst, .. } | Ir::UnaryOp { dst, .. } => {
                known.remove(dst);
            }
            // another path can come here with other values
            Ir::Label(_) => known.clear(),
            _ => {}
        }
    }
}

/// Dead code elimination: remove nop instructions and unreachable code after unconditional jumps.
///
/// A program the VM refuses is left as it is: removing the bad part would make it run.
pub fn dead_code_elimination(program: &mut IrProgram) {
    if check_program(program).is_err() {
        return;
    }
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
///
/// A program the VM refuses is left as it is: removing the bad part would make it run.
pub fn peephole(program: &mut IrProgram) {
    if check_program(program).is_err() {
        return;
    }
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
