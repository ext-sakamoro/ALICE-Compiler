//! Intermediate representation.

use std::fmt;
use std::fmt::Write as _;

use crate::ast::{BinOp, UnaryOp};

/// Register index.
pub type Reg = usize;

/// Label index for jumps.
pub type LabelId = usize;

/// Runtime value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
}

impl Value {
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    #[must_use]
    pub const fn as_float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    #[must_use]
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Int(v) => *v != 0,
            Self::Float(v) => *v != 0.0,
            Self::Bool(v) => *v,
            Self::Str(s) => !s.is_empty(),
            Self::Unit => false,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Str(s) => write!(f, "\"{s}\""),
            Self::Unit => write!(f, "()"),
        }
    }
}

/// IR instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Ir {
    LoadConst {
        dst: Reg,
        val: Value,
    },
    Copy {
        dst: Reg,
        src: Reg,
    },
    BinOp {
        op: BinOp,
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    UnaryOp {
        op: UnaryOp,
        dst: Reg,
        src: Reg,
    },
    Label(LabelId),
    Jump(LabelId),
    JumpIfFalse {
        cond: Reg,
        target: LabelId,
    },
    JumpIfTrue {
        cond: Reg,
        target: LabelId,
    },
    Return(Reg),
    /// No-op (placeholder after dead code elimination).
    Nop,
}

impl fmt::Display for Ir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LoadConst { dst, val } => write!(f, "r{dst} = {val}"),
            Self::Copy { dst, src } => write!(f, "r{dst} = r{src}"),
            Self::BinOp { op, dst, lhs, rhs } => write!(f, "r{dst} = r{lhs} {op} r{rhs}"),
            Self::UnaryOp { op, dst, src } => write!(f, "r{dst} = {op}r{src}"),
            Self::Label(l) => write!(f, "L{l}:"),
            Self::Jump(l) => write!(f, "jump L{l}"),
            Self::JumpIfFalse { cond, target } => {
                write!(f, "jump_if_false r{cond} L{target}")
            }
            Self::JumpIfTrue { cond, target } => write!(f, "jump_if_true r{cond} L{target}"),
            Self::Return(r) => write!(f, "return r{r}"),
            Self::Nop => write!(f, "nop"),
        }
    }
}

/// A compiled IR program.
#[derive(Debug, Clone)]
pub struct IrProgram {
    pub instructions: Vec<Ir>,
    pub register_count: usize,
}

impl IrProgram {
    /// Count the number of non-nop instructions.
    #[must_use]
    pub fn active_instruction_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|ir| !matches!(ir, Ir::Nop))
            .count()
    }

    /// Pretty-print the program.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut out = String::new();
        for (i, ir) in self.instructions.iter().enumerate() {
            let _ = writeln!(out, "{i:4}: {ir}");
        }
        out
    }
}
