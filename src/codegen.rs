//! AST → IR lowering (code generation).

use std::collections::HashMap;

use crate::ast::{BinOp, Expr, UnaryOp};
use crate::ir::{Ir, IrProgram, LabelId, Reg, Value};

/// How deep an expression may nest. There is no outside spec for this; it is chosen because lowering
/// recurses once per level, so an AST as deep as its input can overflow the stack, and 256 levels fit
/// the 2 MiB stack of a spawned thread with room for the caller.
pub const MAX_EXPR_DEPTH: usize = 256;

/// Why an expression has no IR.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompileError {
    /// The expression uses a name no `let` around it binds.
    UndefinedVariable { name: String },
    /// The expression nests deeper than [`MAX_EXPR_DEPTH`].
    TooDeep,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndefinedVariable { name } => write!(f, "undefined variable: {name}"),
            Self::TooDeep => write!(f, "expression nests deeper than {MAX_EXPR_DEPTH}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Lowers AST to IR.
pub struct Codegen {
    instructions: Vec<Ir>,
    pub(crate) next_reg: Reg,
    next_label: LabelId,
    vars: HashMap<String, Reg>,
    depth: usize,
}

impl Default for Codegen {
    fn default() -> Self {
        Self::new()
    }
}

impl Codegen {
    #[must_use]
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            next_reg: 0,
            next_label: 0,
            vars: HashMap::new(),
            depth: 0,
        }
    }

    const fn alloc_reg(&mut self) -> Reg {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    const fn alloc_label(&mut self) -> LabelId {
        let l = self.next_label;
        self.next_label += 1;
        l
    }

    fn emit(&mut self, ir: Ir) {
        self.instructions.push(ir);
    }

    /// Compile an AST expression into IR, returning the result register.
    ///
    /// # Errors
    ///
    /// [`CompileError::UndefinedVariable`] if an undefined variable is referenced,
    /// [`CompileError::TooDeep`] past [`MAX_EXPR_DEPTH`].
    pub fn compile_expr(&mut self, expr: &Expr) -> Result<Reg, CompileError> {
        if self.depth >= MAX_EXPR_DEPTH {
            return Err(CompileError::TooDeep);
        }
        self.depth += 1;
        let reg = self.compile_node(expr);
        self.depth -= 1;
        reg
    }

    fn compile_node(&mut self, expr: &Expr) -> Result<Reg, CompileError> {
        match expr {
            Expr::Int(v) => Ok(self.emit_const(Value::Int(*v))),
            Expr::Float(v) => Ok(self.emit_const(Value::Float(*v))),
            Expr::Bool(v) => Ok(self.emit_const(Value::Bool(*v))),
            Expr::Str(s) => Ok(self.emit_const(Value::Str(s.clone()))),
            Expr::Var(name) => self.compile_var(name),
            Expr::Binary { op, lhs, rhs } => self.compile_binary(*op, lhs, rhs),
            Expr::Unary { op, operand } => self.compile_unary(*op, operand),
            Expr::Let { name, value, body } => self.compile_let(name, value, body),
            Expr::If { cond, then, else_ } => self.compile_if(cond, then, else_),
            Expr::Fn { body, .. } => self.compile_expr(body),
            Expr::Call { func, args: _ } => self.compile_expr(func),
            Expr::Block(exprs) => self.compile_block(exprs),
        }
    }

    fn emit_const(&mut self, val: Value) -> Reg {
        let dst = self.alloc_reg();
        self.emit(Ir::LoadConst { dst, val });
        dst
    }

    fn compile_var(&mut self, name: &str) -> Result<Reg, CompileError> {
        let src = self
            .vars
            .get(name)
            .copied()
            .ok_or_else(|| CompileError::UndefinedVariable {
                name: name.to_owned(),
            })?;
        let dst = self.alloc_reg();
        self.emit(Ir::Copy { dst, src });
        Ok(dst)
    }

    fn compile_binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Result<Reg, CompileError> {
        let lhs_reg = self.compile_expr(lhs)?;
        let rhs_reg = self.compile_expr(rhs)?;
        let dst = self.alloc_reg();
        self.emit(Ir::BinOp {
            op,
            dst,
            lhs: lhs_reg,
            rhs: rhs_reg,
        });
        Ok(dst)
    }

    fn compile_unary(&mut self, op: UnaryOp, operand: &Expr) -> Result<Reg, CompileError> {
        let src = self.compile_expr(operand)?;
        let dst = self.alloc_reg();
        self.emit(Ir::UnaryOp { op, dst, src });
        Ok(dst)
    }

    fn compile_let(&mut self, name: &str, value: &Expr, body: &Expr) -> Result<Reg, CompileError> {
        let val_reg = self.compile_expr(value)?;
        let old = self.vars.insert(name.to_owned(), val_reg);
        let body_reg = self.compile_expr(body)?;
        if let Some(prev) = old {
            self.vars.insert(name.to_owned(), prev);
        } else {
            self.vars.remove(name);
        }
        Ok(body_reg)
    }

    fn compile_if(&mut self, cond: &Expr, then: &Expr, else_: &Expr) -> Result<Reg, CompileError> {
        let cond_reg = self.compile_expr(cond)?;
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let result = self.alloc_reg();

        self.emit(Ir::JumpIfFalse {
            cond: cond_reg,
            target: else_label,
        });

        let then_reg = self.compile_expr(then)?;
        self.emit(Ir::Copy {
            dst: result,
            src: then_reg,
        });
        self.emit(Ir::Jump(end_label));

        self.emit(Ir::Label(else_label));
        let else_reg = self.compile_expr(else_)?;
        self.emit(Ir::Copy {
            dst: result,
            src: else_reg,
        });

        self.emit(Ir::Label(end_label));
        Ok(result)
    }

    fn compile_block(&mut self, exprs: &[Expr]) -> Result<Reg, CompileError> {
        let mut last = self.alloc_reg();
        self.emit(Ir::LoadConst {
            dst: last,
            val: Value::Unit,
        });
        for e in exprs {
            last = self.compile_expr(e)?;
        }
        Ok(last)
    }

    /// Compile a top-level expression and produce an `IrProgram`.
    ///
    /// # Errors
    ///
    /// [`CompileError::UndefinedVariable`] if an undefined variable is referenced,
    /// [`CompileError::TooDeep`] past [`MAX_EXPR_DEPTH`].
    pub fn compile(mut self, expr: &Expr) -> Result<IrProgram, CompileError> {
        let result = self.compile_expr(expr)?;
        self.emit(Ir::Return(result));
        Ok(IrProgram {
            register_count: self.next_reg,
            instructions: self.instructions,
        })
    }
}
