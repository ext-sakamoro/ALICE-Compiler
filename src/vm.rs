//! Register-based VM interpreter for IR.

use std::collections::HashMap;

use crate::eval::{eval_binop, eval_unaryop};
use crate::ir::{Ir, IrProgram, LabelId, Value};

/// Simple register-based VM to execute IR programs.
pub struct Vm {
    pub(crate) registers: Vec<Value>,
}

impl Default for Vm {
    fn default() -> Self {
        Self::new()
    }
}

impl Vm {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registers: Vec::new(),
        }
    }

    /// Execute an IR program and return the result.
    ///
    /// # Errors
    ///
    /// Returns an error string if execution fails.
    pub fn execute(&mut self, program: &IrProgram) -> Result<Value, String> {
        self.registers = vec![Value::Unit; program.register_count];

        let mut label_map: HashMap<LabelId, usize> = HashMap::new();
        for (i, ir) in program.instructions.iter().enumerate() {
            if let Ir::Label(l) = ir {
                label_map.insert(*l, i);
            }
        }

        let mut pc = 0;
        let len = program.instructions.len();

        while pc < len {
            match &program.instructions[pc] {
                Ir::LoadConst { dst, val } => {
                    self.registers[*dst] = val.clone();
                }
                Ir::Copy { dst, src } => {
                    self.registers[*dst] = self.registers[*src].clone();
                }
                Ir::BinOp { op, dst, lhs, rhs } => {
                    let result = eval_binop(*op, &self.registers[*lhs], &self.registers[*rhs])
                        .ok_or_else(|| {
                            format!(
                                "cannot apply {op} to {:?} and {:?}",
                                self.registers[*lhs], self.registers[*rhs]
                            )
                        })?;
                    self.registers[*dst] = result;
                }
                Ir::UnaryOp { op, dst, src } => {
                    let result = eval_unaryop(*op, &self.registers[*src]).ok_or_else(|| {
                        format!("cannot apply {op} to {:?}", self.registers[*src])
                    })?;
                    self.registers[*dst] = result;
                }
                Ir::Label(_) | Ir::Nop => {}
                Ir::Jump(target) => {
                    pc = label_map[target];
                    continue;
                }
                Ir::JumpIfFalse { cond, target } => {
                    if !self.registers[*cond].is_truthy() {
                        pc = label_map[target];
                        continue;
                    }
                }
                Ir::JumpIfTrue { cond, target } => {
                    if self.registers[*cond].is_truthy() {
                        pc = label_map[target];
                        continue;
                    }
                }
                Ir::Return(reg) => {
                    return Ok(self.registers[*reg].clone());
                }
            }
            pc += 1;
        }

        Ok(Value::Unit)
    }
}
