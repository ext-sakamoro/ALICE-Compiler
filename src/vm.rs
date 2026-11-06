//! Register-based VM interpreter for IR.

use std::collections::HashMap;

use crate::eval::{eval_binop, eval_unaryop};
use crate::ir::{Ir, IrProgram, LabelId, Reg, Value};

/// What one execution may use.
///
/// All of it is counted, never timed, so a program gives the same result
/// on every machine. There is no outside spec for the defaults; each says why it was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmLimits {
    /// Instructions executed. Default 2^24: a program has no other way out of `Label 0; Jump 0`, and
    /// this is long enough for what a compiled expression does, which has no backward jumps.
    pub max_steps: u64,
    /// Bytes of strings made, by `+`, by a copy and by a load, added up over the execution, freed or
    /// not. Default 64 MiB: with the per-string cap it bounds what registers can hold at once.
    pub max_string_bytes: usize,
    /// `register_count` of a program. Default 2^20: a register costs its size whether or not the
    /// program uses it, and a compiled expression needs one per node.
    pub max_registers: usize,
}

impl VmLimits {
    /// The defaults.
    pub const DEFAULT: Self = Self {
        max_steps: 1 << 24,
        max_string_bytes: 64 << 20,
        max_registers: 1 << 20,
    };
}

impl Default for VmLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Simple register-based VM to execute IR programs.
pub struct Vm {
    pub(crate) registers: Vec<Value>,
    limits: VmLimits,
}

impl Default for Vm {
    fn default() -> Self {
        Self::new()
    }
}

impl Vm {
    #[must_use]
    pub const fn new() -> Self {
        Self::with_limits(VmLimits::DEFAULT)
    }

    #[must_use]
    pub const fn with_limits(limits: VmLimits) -> Self {
        Self {
            registers: Vec::new(),
            limits,
        }
    }

    /// Adds a value about to be made to the string bytes of this execution.
    fn charge(&self, spent: &mut usize, val: &Value) -> Result<(), String> {
        if let Value::Str(s) = val {
            *spent = spent.saturating_add(s.len());
            if *spent > self.limits.max_string_bytes {
                return Err(format!(
                    "string bytes over the limit of {}",
                    self.limits.max_string_bytes
                ));
            }
        }
        Ok(())
    }

    /// Execute an IR program and return the result.
    ///
    /// # Errors
    ///
    /// Returns an error string if the program uses a register outside `register_count` or a label
    /// it does not define (found before any instruction runs, on any path), if it goes over
    /// [`VmLimits`], if the registers cannot be allocated, or if execution fails.
    pub fn execute(&mut self, program: &IrProgram) -> Result<Value, String> {
        let label_map = check_program(program)?;

        if program.register_count > self.limits.max_registers {
            return Err(format!(
                "{} registers is over the limit of {}",
                program.register_count, self.limits.max_registers
            ));
        }
        let mut registers = Vec::new();
        registers
            .try_reserve_exact(program.register_count)
            .map_err(|_| format!("cannot allocate {} registers", program.register_count))?;
        registers.resize(program.register_count, Value::Unit);
        self.registers = registers;

        let mut pc = 0;
        let len = program.instructions.len();
        let mut steps = 0u64;
        let mut string_bytes = 0usize;

        while pc < len {
            if steps >= self.limits.max_steps {
                return Err(format!("steps over the limit of {}", self.limits.max_steps));
            }
            steps += 1;
            match &program.instructions[pc] {
                Ir::LoadConst { dst, val } => {
                    self.charge(&mut string_bytes, val)?;
                    self.registers[*dst] = val.clone();
                }
                Ir::Copy { dst, src } => {
                    self.charge(&mut string_bytes, &self.registers[*src])?;
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
                    self.charge(&mut string_bytes, &result)?;
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

/// Every register and jump target of the program exists. Returns where each label is.
pub(crate) fn check_program(program: &IrProgram) -> Result<HashMap<LabelId, usize>, String> {
    let mut labels: HashMap<LabelId, usize> = HashMap::new();
    for (i, ir) in program.instructions.iter().enumerate() {
        if let Ir::Label(l) = ir {
            labels.insert(*l, i);
        }
    }
    let register = |r: Reg| {
        if r < program.register_count {
            Ok(())
        } else {
            Err(format!(
                "register {r} is outside the {} of the program",
                program.register_count
            ))
        }
    };
    let label = |l: &LabelId| {
        if labels.contains_key(l) {
            Ok(())
        } else {
            Err(format!("label {l} is not defined"))
        }
    };
    for ir in &program.instructions {
        match ir {
            Ir::LoadConst { dst, .. } => register(*dst)?,
            Ir::Copy { dst, src } | Ir::UnaryOp { dst, src, .. } => {
                register(*dst)?;
                register(*src)?;
            }
            Ir::BinOp { dst, lhs, rhs, .. } => {
                register(*dst)?;
                register(*lhs)?;
                register(*rhs)?;
            }
            Ir::Label(_) | Ir::Nop => {}
            Ir::Jump(target) => label(target)?,
            Ir::JumpIfFalse { cond, target } | Ir::JumpIfTrue { cond, target } => {
                register(*cond)?;
                label(target)?;
            }
            Ir::Return(reg) => register(*reg)?,
        }
    }
    Ok(labels)
}
