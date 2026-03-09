#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

//! ALICE-Compiler: DSL/JIT compiler infrastructure.
//!
//! Provides AST nodes, IR (intermediate representation), code generation,
//! optimization passes, and a register-based VM for execution.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::And => "&&",
            Self::Or => "||",
        };
        write!(f, "{s}")
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Neg => "-",
            Self::Not => "!",
        };
        write!(f, "{s}")
    }
}

/// AST expression node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Integer literal.
    Int(i64),
    /// Floating-point literal.
    Float(f64),
    /// Boolean literal.
    Bool(bool),
    /// String literal.
    Str(String),
    /// Variable reference.
    Var(String),
    /// Binary operation.
    Binary {
        op: BinOp,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    /// Unary operation.
    Unary { op: UnaryOp, operand: Box<Self> },
    /// Let binding: `let name = value; body`
    Let {
        name: String,
        value: Box<Self>,
        body: Box<Self>,
    },
    /// If expression: `if cond then else_`
    If {
        cond: Box<Self>,
        then: Box<Self>,
        else_: Box<Self>,
    },
    /// Function definition (lambda): `fn(params) -> body`
    Fn {
        name: Option<String>,
        params: Vec<String>,
        body: Box<Self>,
    },
    /// Function call.
    Call { func: Box<Self>, args: Vec<Self> },
    /// Block: sequence of expressions, value is last.
    Block(Vec<Self>),
}

impl Expr {
    /// Create a binary expression.
    #[must_use]
    pub fn binary(op: BinOp, lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Create a unary expression.
    #[must_use]
    pub fn unary(op: UnaryOp, operand: Self) -> Self {
        Self::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    /// Create a let binding.
    #[must_use]
    pub fn let_bind(name: impl Into<String>, value: Self, body: Self) -> Self {
        Self::Let {
            name: name.into(),
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    /// Create an if expression.
    #[must_use]
    pub fn if_expr(cond: Self, then: Self, else_: Self) -> Self {
        Self::If {
            cond: Box::new(cond),
            then: Box::new(then),
            else_: Box::new(else_),
        }
    }

    /// Create a function call.
    #[must_use]
    pub fn call(func: Self, args: Vec<Self>) -> Self {
        Self::Call {
            func: Box::new(func),
            args,
        }
    }

    /// Check if the expression is a literal value.
    #[must_use]
    pub const fn is_literal(&self) -> bool {
        matches!(
            self,
            Self::Int(_) | Self::Float(_) | Self::Bool(_) | Self::Str(_)
        )
    }

    /// Count the number of AST nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            Self::Int(_) | Self::Float(_) | Self::Bool(_) | Self::Str(_) | Self::Var(_) => 1,
            Self::Binary { lhs, rhs, .. } => 1 + lhs.node_count() + rhs.node_count(),
            Self::Unary { operand, .. } => 1 + operand.node_count(),
            Self::Let { value, body, .. } => 1 + value.node_count() + body.node_count(),
            Self::If {
                cond, then, else_, ..
            } => 1 + cond.node_count() + then.node_count() + else_.node_count(),
            Self::Fn { body, .. } => 1 + body.node_count(),
            Self::Call { func, args } => {
                1 + func.node_count() + args.iter().map(Self::node_count).sum::<usize>()
            }
            Self::Block(exprs) => 1 + exprs.iter().map(Self::node_count).sum::<usize>(),
        }
    }
}

// ---------------------------------------------------------------------------
// IR
// ---------------------------------------------------------------------------

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
    /// Try to extract an integer.
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract a float.
    #[must_use]
    pub const fn as_float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract a boolean.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Check if the value is truthy.
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
    /// Load a constant value into a register.
    LoadConst { dst: Reg, val: Value },
    /// Copy one register to another.
    Copy { dst: Reg, src: Reg },
    /// Binary operation: dst = lhs op rhs.
    BinOp {
        op: BinOp,
        dst: Reg,
        lhs: Reg,
        rhs: Reg,
    },
    /// Unary operation: dst = op operand.
    UnaryOp { op: UnaryOp, dst: Reg, src: Reg },
    /// Define a label.
    Label(LabelId),
    /// Unconditional jump.
    Jump(LabelId),
    /// Jump if register value is falsy.
    JumpIfFalse { cond: Reg, target: LabelId },
    /// Jump if register value is truthy.
    JumpIfTrue { cond: Reg, target: LabelId },
    /// Return a value from the current function.
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

// ---------------------------------------------------------------------------
// Code Generation (AST -> IR lowering)
// ---------------------------------------------------------------------------

/// Lowers AST to IR.
pub struct Codegen {
    instructions: Vec<Ir>,
    next_reg: Reg,
    next_label: LabelId,
    vars: HashMap<String, Reg>,
}

impl Default for Codegen {
    fn default() -> Self {
        Self::new()
    }
}

impl Codegen {
    /// Create a new code generator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            next_reg: 0,
            next_label: 0,
            vars: HashMap::new(),
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
    /// # Panics
    ///
    /// Panics if an undefined variable is referenced.
    pub fn compile_expr(&mut self, expr: &Expr) -> Reg {
        match expr {
            Expr::Int(v) => self.emit_const(Value::Int(*v)),
            Expr::Float(v) => self.emit_const(Value::Float(*v)),
            Expr::Bool(v) => self.emit_const(Value::Bool(*v)),
            Expr::Str(s) => self.emit_const(Value::Str(s.clone())),
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

    fn compile_var(&mut self, name: &str) -> Reg {
        let src = self.vars.get(name).copied().unwrap_or_else(|| {
            panic!("undefined variable: {name}");
        });
        let dst = self.alloc_reg();
        self.emit(Ir::Copy { dst, src });
        dst
    }

    fn compile_binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Reg {
        let lhs_reg = self.compile_expr(lhs);
        let rhs_reg = self.compile_expr(rhs);
        let dst = self.alloc_reg();
        self.emit(Ir::BinOp {
            op,
            dst,
            lhs: lhs_reg,
            rhs: rhs_reg,
        });
        dst
    }

    fn compile_unary(&mut self, op: UnaryOp, operand: &Expr) -> Reg {
        let src = self.compile_expr(operand);
        let dst = self.alloc_reg();
        self.emit(Ir::UnaryOp { op, dst, src });
        dst
    }

    fn compile_let(&mut self, name: &str, value: &Expr, body: &Expr) -> Reg {
        let val_reg = self.compile_expr(value);
        let old = self.vars.insert(name.to_owned(), val_reg);
        let body_reg = self.compile_expr(body);
        if let Some(prev) = old {
            self.vars.insert(name.to_owned(), prev);
        } else {
            self.vars.remove(name);
        }
        body_reg
    }

    fn compile_if(&mut self, cond: &Expr, then: &Expr, else_: &Expr) -> Reg {
        let cond_reg = self.compile_expr(cond);
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let result = self.alloc_reg();

        self.emit(Ir::JumpIfFalse {
            cond: cond_reg,
            target: else_label,
        });

        let then_reg = self.compile_expr(then);
        self.emit(Ir::Copy {
            dst: result,
            src: then_reg,
        });
        self.emit(Ir::Jump(end_label));

        self.emit(Ir::Label(else_label));
        let else_reg = self.compile_expr(else_);
        self.emit(Ir::Copy {
            dst: result,
            src: else_reg,
        });

        self.emit(Ir::Label(end_label));
        result
    }

    fn compile_block(&mut self, exprs: &[Expr]) -> Reg {
        let mut last = self.alloc_reg();
        self.emit(Ir::LoadConst {
            dst: last,
            val: Value::Unit,
        });
        for e in exprs {
            last = self.compile_expr(e);
        }
        last
    }

    /// Compile a top-level expression and produce an `IrProgram`.
    #[must_use]
    pub fn compile(mut self, expr: &Expr) -> IrProgram {
        let result = self.compile_expr(expr);
        self.emit(Ir::Return(result));
        IrProgram {
            register_count: self.next_reg,
            instructions: self.instructions,
        }
    }
}

// ---------------------------------------------------------------------------
// Optimization Passes
// ---------------------------------------------------------------------------

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
        // Copy to self is a nop.
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

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

fn eval_binop(op: BinOp, lhs: &Value, rhs: &Value) -> Option<Value> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => eval_binop_int(op, *a, *b),
        (Value::Float(a), Value::Float(b)) => eval_binop_float(op, *a, *b),
        (Value::Bool(a), Value::Bool(b)) => eval_binop_bool(op, *a, *b),
        (Value::Str(a), Value::Str(b)) if op == BinOp::Add => Some(Value::Str(format!("{a}{b}"))),
        _ => None,
    }
}

const fn eval_binop_int(op: BinOp, a: i64, b: i64) -> Option<Value> {
    Some(match op {
        BinOp::Add => Value::Int(a.wrapping_add(b)),
        BinOp::Sub => Value::Int(a.wrapping_sub(b)),
        BinOp::Mul => Value::Int(a.wrapping_mul(b)),
        BinOp::Div => {
            if b == 0 {
                return None;
            }
            Value::Int(a / b)
        }
        BinOp::Mod => {
            if b == 0 {
                return None;
            }
            Value::Int(a % b)
        }
        BinOp::Eq => Value::Bool(a == b),
        BinOp::Ne => Value::Bool(a != b),
        BinOp::Lt => Value::Bool(a < b),
        BinOp::Le => Value::Bool(a <= b),
        BinOp::Gt => Value::Bool(a > b),
        BinOp::Ge => Value::Bool(a >= b),
        BinOp::And | BinOp::Or => return None,
    })
}

fn eval_binop_float(op: BinOp, a: f64, b: f64) -> Option<Value> {
    Some(match op {
        BinOp::Add => Value::Float(a + b),
        BinOp::Sub => Value::Float(a - b),
        BinOp::Mul => Value::Float(a * b),
        BinOp::Div => Value::Float(a / b),
        BinOp::Mod => Value::Float(a % b),
        BinOp::Eq => Value::Bool((a - b).abs() < f64::EPSILON),
        BinOp::Ne => Value::Bool((a - b).abs() >= f64::EPSILON),
        BinOp::Lt => Value::Bool(a < b),
        BinOp::Le => Value::Bool(a <= b),
        BinOp::Gt => Value::Bool(a > b),
        BinOp::Ge => Value::Bool(a >= b),
        BinOp::And | BinOp::Or => return None,
    })
}

const fn eval_binop_bool(op: BinOp, a: bool, b: bool) -> Option<Value> {
    Some(match op {
        BinOp::And => Value::Bool(a && b),
        BinOp::Or => Value::Bool(a || b),
        BinOp::Eq => Value::Bool(a == b),
        BinOp::Ne => Value::Bool(a != b),
        _ => return None,
    })
}

const fn eval_unaryop(op: UnaryOp, val: &Value) -> Option<Value> {
    match (op, val) {
        (UnaryOp::Neg, Value::Int(v)) => Some(Value::Int(-*v)),
        (UnaryOp::Neg, Value::Float(v)) => Some(Value::Float(-*v)),
        (UnaryOp::Not, Value::Bool(v)) => Some(Value::Bool(!*v)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// VM (register-based interpreter for IR)
// ---------------------------------------------------------------------------

/// Simple register-based VM to execute IR programs.
pub struct Vm {
    registers: Vec<Value>,
}

impl Default for Vm {
    fn default() -> Self {
        Self::new()
    }
}

impl Vm {
    /// Create a new VM.
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

// ---------------------------------------------------------------------------
// Convenience: compile and run
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === AST construction tests ===

    #[test]
    fn test_int_literal() {
        let e = Expr::Int(42);
        assert!(e.is_literal());
        assert_eq!(e.node_count(), 1);
    }

    #[test]
    fn test_float_literal() {
        let e = Expr::Float(3.0);
        assert!(e.is_literal());
    }

    #[test]
    fn test_bool_literal() {
        let e = Expr::Bool(true);
        assert!(e.is_literal());
    }

    #[test]
    fn test_string_literal() {
        let e = Expr::Str("hello".into());
        assert!(e.is_literal());
    }

    #[test]
    fn test_var_not_literal() {
        let e = Expr::Var("x".into());
        assert!(!e.is_literal());
    }

    #[test]
    fn test_binary_construction() {
        let e = Expr::binary(BinOp::Add, Expr::Int(1), Expr::Int(2));
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn test_unary_construction() {
        let e = Expr::unary(UnaryOp::Neg, Expr::Int(5));
        assert_eq!(e.node_count(), 2);
    }

    #[test]
    fn test_let_construction() {
        let e = Expr::let_bind("x", Expr::Int(10), Expr::Var("x".into()));
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn test_if_construction() {
        let e = Expr::if_expr(Expr::Bool(true), Expr::Int(1), Expr::Int(0));
        assert_eq!(e.node_count(), 4);
    }

    #[test]
    fn test_block_construction() {
        let e = Expr::Block(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]);
        assert_eq!(e.node_count(), 4);
    }

    #[test]
    fn test_call_construction() {
        let e = Expr::call(Expr::Var("f".into()), vec![Expr::Int(1)]);
        assert_eq!(e.node_count(), 3);
    }

    #[test]
    fn test_fn_construction() {
        let e = Expr::Fn {
            name: Some("add".into()),
            params: vec!["a".into(), "b".into()],
            body: Box::new(Expr::binary(
                BinOp::Add,
                Expr::Var("a".into()),
                Expr::Var("b".into()),
            )),
        };
        assert_eq!(e.node_count(), 4);
    }

    #[test]
    fn test_nested_binary() {
        let e = Expr::binary(
            BinOp::Mul,
            Expr::binary(BinOp::Add, Expr::Int(1), Expr::Int(2)),
            Expr::Int(3),
        );
        assert_eq!(e.node_count(), 5);
    }

    // === BinOp Display ===

    #[test]
    fn test_binop_display() {
        assert_eq!(format!("{}", BinOp::Add), "+");
        assert_eq!(format!("{}", BinOp::Sub), "-");
        assert_eq!(format!("{}", BinOp::Mul), "*");
        assert_eq!(format!("{}", BinOp::Div), "/");
        assert_eq!(format!("{}", BinOp::Mod), "%");
        assert_eq!(format!("{}", BinOp::Eq), "==");
        assert_eq!(format!("{}", BinOp::Ne), "!=");
        assert_eq!(format!("{}", BinOp::Lt), "<");
        assert_eq!(format!("{}", BinOp::Le), "<=");
        assert_eq!(format!("{}", BinOp::Gt), ">");
        assert_eq!(format!("{}", BinOp::Ge), ">=");
        assert_eq!(format!("{}", BinOp::And), "&&");
        assert_eq!(format!("{}", BinOp::Or), "||");
    }

    #[test]
    fn test_unaryop_display() {
        assert_eq!(format!("{}", UnaryOp::Neg), "-");
        assert_eq!(format!("{}", UnaryOp::Not), "!");
    }

    // === Value tests ===

    #[test]
    fn test_value_as_int() {
        assert_eq!(Value::Int(42).as_int(), Some(42));
        assert_eq!(Value::Float(1.0).as_int(), None);
    }

    #[test]
    fn test_value_as_float() {
        assert_eq!(Value::Float(2.719).as_float(), Some(2.719));
        assert_eq!(Value::Int(1).as_float(), None);
    }

    #[test]
    fn test_value_as_bool() {
        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(Value::Int(1).as_bool(), None);
    }

    #[test]
    fn test_value_truthy_int() {
        assert!(Value::Int(1).is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(Value::Int(-1).is_truthy());
    }

    #[test]
    fn test_value_truthy_float() {
        assert!(Value::Float(0.1).is_truthy());
        assert!(!Value::Float(0.0).is_truthy());
    }

    #[test]
    fn test_value_truthy_bool() {
        assert!(Value::Bool(true).is_truthy());
        assert!(!Value::Bool(false).is_truthy());
    }

    #[test]
    fn test_value_truthy_str() {
        assert!(Value::Str("hello".into()).is_truthy());
        assert!(!Value::Str(String::new()).is_truthy());
    }

    #[test]
    fn test_value_truthy_unit() {
        assert!(!Value::Unit.is_truthy());
    }

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value::Int(42)), "42");
        assert_eq!(format!("{}", Value::Bool(true)), "true");
        assert_eq!(format!("{}", Value::Str("hi".into())), "\"hi\"");
        assert_eq!(format!("{}", Value::Unit), "()");
    }

    // === IR Display ===

    #[test]
    fn test_ir_display_load_const() {
        let ir = Ir::LoadConst {
            dst: 0,
            val: Value::Int(42),
        };
        assert_eq!(format!("{ir}"), "r0 = 42");
    }

    #[test]
    fn test_ir_display_binop() {
        let ir = Ir::BinOp {
            op: BinOp::Add,
            dst: 2,
            lhs: 0,
            rhs: 1,
        };
        assert_eq!(format!("{ir}"), "r2 = r0 + r1");
    }

    #[test]
    fn test_ir_display_copy() {
        let ir = Ir::Copy { dst: 1, src: 0 };
        assert_eq!(format!("{ir}"), "r1 = r0");
    }

    #[test]
    fn test_ir_display_label() {
        assert_eq!(format!("{}", Ir::Label(3)), "L3:");
    }

    #[test]
    fn test_ir_display_jump() {
        assert_eq!(format!("{}", Ir::Jump(5)), "jump L5");
    }

    #[test]
    fn test_ir_display_jump_if_false() {
        let ir = Ir::JumpIfFalse { cond: 0, target: 1 };
        assert_eq!(format!("{ir}"), "jump_if_false r0 L1");
    }

    #[test]
    fn test_ir_display_jump_if_true() {
        let ir = Ir::JumpIfTrue { cond: 0, target: 1 };
        assert_eq!(format!("{ir}"), "jump_if_true r0 L1");
    }

    #[test]
    fn test_ir_display_return() {
        assert_eq!(format!("{}", Ir::Return(0)), "return r0");
    }

    #[test]
    fn test_ir_display_nop() {
        assert_eq!(format!("{}", Ir::Nop), "nop");
    }

    #[test]
    fn test_ir_display_unaryop() {
        let ir = Ir::UnaryOp {
            op: UnaryOp::Neg,
            dst: 1,
            src: 0,
        };
        assert_eq!(format!("{ir}"), "r1 = -r0");
    }

    // === Codegen tests ===

    #[test]
    fn test_codegen_int() {
        let program = Codegen::new().compile(&Expr::Int(42));
        assert!(program.register_count >= 1);
        assert!(!program.instructions.is_empty());
    }

    #[test]
    fn test_codegen_binary() {
        let e = Expr::binary(BinOp::Add, Expr::Int(1), Expr::Int(2));
        let program = Codegen::new().compile(&e);
        assert!(program.register_count >= 3);
    }

    #[test]
    fn test_codegen_let() {
        let e = Expr::let_bind("x", Expr::Int(10), Expr::Var("x".into()));
        let program = Codegen::new().compile(&e);
        assert!(program.register_count >= 2);
    }

    #[test]
    fn test_codegen_if() {
        let e = Expr::if_expr(Expr::Bool(true), Expr::Int(1), Expr::Int(0));
        let program = Codegen::new().compile(&e);
        assert!(program
            .instructions
            .iter()
            .any(|ir| matches!(ir, Ir::JumpIfFalse { .. })));
    }

    #[test]
    fn test_codegen_block() {
        let e = Expr::Block(vec![Expr::Int(1), Expr::Int(2)]);
        let program = Codegen::new().compile(&e);
        assert!(program.register_count >= 3);
    }

    #[test]
    fn test_codegen_unary() {
        let e = Expr::unary(UnaryOp::Neg, Expr::Int(5));
        let program = Codegen::new().compile(&e);
        assert!(program
            .instructions
            .iter()
            .any(|ir| matches!(ir, Ir::UnaryOp { .. })));
    }

    #[test]
    fn test_codegen_default() {
        let cg = Codegen::default();
        assert_eq!(cg.next_reg, 0);
    }

    // === VM execution tests ===

    #[test]
    fn test_vm_int() {
        let result = compile_and_run(&Expr::Int(42)).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_vm_float() {
        let result = compile_and_run(&Expr::Float(2.5)).unwrap();
        assert_eq!(result, Value::Float(2.5));
    }

    #[test]
    fn test_vm_bool() {
        let result = compile_and_run(&Expr::Bool(false)).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_vm_string() {
        let result = compile_and_run(&Expr::Str("hello".into())).unwrap();
        assert_eq!(result, Value::Str("hello".into()));
    }

    #[test]
    fn test_vm_add_int() {
        let e = Expr::binary(BinOp::Add, Expr::Int(3), Expr::Int(4));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(7));
    }

    #[test]
    fn test_vm_sub_int() {
        let e = Expr::binary(BinOp::Sub, Expr::Int(10), Expr::Int(3));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(7));
    }

    #[test]
    fn test_vm_mul_int() {
        let e = Expr::binary(BinOp::Mul, Expr::Int(6), Expr::Int(7));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(42));
    }

    #[test]
    fn test_vm_div_int() {
        let e = Expr::binary(BinOp::Div, Expr::Int(10), Expr::Int(3));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(3));
    }

    #[test]
    fn test_vm_mod_int() {
        let e = Expr::binary(BinOp::Mod, Expr::Int(10), Expr::Int(3));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(1));
    }

    #[test]
    fn test_vm_eq_true() {
        let e = Expr::binary(BinOp::Eq, Expr::Int(5), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_eq_false() {
        let e = Expr::binary(BinOp::Eq, Expr::Int(5), Expr::Int(3));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_vm_ne() {
        let e = Expr::binary(BinOp::Ne, Expr::Int(5), Expr::Int(3));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_lt() {
        let e = Expr::binary(BinOp::Lt, Expr::Int(3), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_le() {
        let e = Expr::binary(BinOp::Le, Expr::Int(5), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_gt() {
        let e = Expr::binary(BinOp::Gt, Expr::Int(5), Expr::Int(3));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_ge() {
        let e = Expr::binary(BinOp::Ge, Expr::Int(3), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_vm_and() {
        let e = Expr::binary(BinOp::And, Expr::Bool(true), Expr::Bool(false));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_vm_or() {
        let e = Expr::binary(BinOp::Or, Expr::Bool(false), Expr::Bool(true));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_neg() {
        let e = Expr::unary(UnaryOp::Neg, Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(-5));
    }

    #[test]
    fn test_vm_not() {
        let e = Expr::unary(UnaryOp::Not, Expr::Bool(true));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_vm_neg_float() {
        let e = Expr::unary(UnaryOp::Neg, Expr::Float(3.5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Float(-3.5));
    }

    #[test]
    fn test_vm_add_float() {
        let e = Expr::binary(BinOp::Add, Expr::Float(1.5), Expr::Float(2.5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Float(4.0));
    }

    #[test]
    fn test_vm_sub_float() {
        let e = Expr::binary(BinOp::Sub, Expr::Float(5.0), Expr::Float(2.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Float(3.0));
    }

    #[test]
    fn test_vm_mul_float() {
        let e = Expr::binary(BinOp::Mul, Expr::Float(3.0), Expr::Float(4.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Float(12.0));
    }

    #[test]
    fn test_vm_div_float() {
        let e = Expr::binary(BinOp::Div, Expr::Float(10.0), Expr::Float(4.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Float(2.5));
    }

    #[test]
    fn test_vm_string_concat() {
        let e = Expr::binary(
            BinOp::Add,
            Expr::Str("hello ".into()),
            Expr::Str("world".into()),
        );
        assert_eq!(
            compile_and_run(&e).unwrap(),
            Value::Str("hello world".into())
        );
    }

    #[test]
    fn test_vm_let_binding() {
        let e = Expr::let_bind("x", Expr::Int(10), Expr::Var("x".into()));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(10));
    }

    #[test]
    fn test_vm_let_with_computation() {
        let e = Expr::let_bind(
            "x",
            Expr::Int(5),
            Expr::binary(BinOp::Add, Expr::Var("x".into()), Expr::Int(3)),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(8));
    }

    #[test]
    fn test_vm_nested_let() {
        let e = Expr::let_bind(
            "x",
            Expr::Int(5),
            Expr::let_bind(
                "y",
                Expr::Int(10),
                Expr::binary(BinOp::Add, Expr::Var("x".into()), Expr::Var("y".into())),
            ),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(15));
    }

    #[test]
    fn test_vm_shadowed_let() {
        let e = Expr::let_bind(
            "x",
            Expr::Int(5),
            Expr::let_bind("x", Expr::Int(10), Expr::Var("x".into())),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(10));
    }

    #[test]
    fn test_vm_if_true() {
        let e = Expr::if_expr(Expr::Bool(true), Expr::Int(1), Expr::Int(0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(1));
    }

    #[test]
    fn test_vm_if_false() {
        let e = Expr::if_expr(Expr::Bool(false), Expr::Int(1), Expr::Int(0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(0));
    }

    #[test]
    fn test_vm_if_with_comparison() {
        let e = Expr::if_expr(
            Expr::binary(BinOp::Lt, Expr::Int(3), Expr::Int(5)),
            Expr::Int(100),
            Expr::Int(200),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(100));
    }

    #[test]
    fn test_vm_nested_if() {
        let e = Expr::if_expr(
            Expr::Bool(true),
            Expr::if_expr(Expr::Bool(false), Expr::Int(1), Expr::Int(2)),
            Expr::Int(3),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(2));
    }

    #[test]
    fn test_vm_block_returns_last() {
        let e = Expr::Block(vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]);
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(3));
    }

    #[test]
    fn test_vm_empty_block() {
        let e = Expr::Block(vec![]);
        assert_eq!(compile_and_run(&e).unwrap(), Value::Unit);
    }

    #[test]
    fn test_vm_complex_expr() {
        let e = Expr::let_bind(
            "a",
            Expr::Int(10),
            Expr::let_bind(
                "b",
                Expr::Int(20),
                Expr::if_expr(
                    Expr::binary(BinOp::Lt, Expr::Var("a".into()), Expr::Var("b".into())),
                    Expr::binary(BinOp::Mul, Expr::Var("a".into()), Expr::Int(2)),
                    Expr::binary(BinOp::Mul, Expr::Var("b".into()), Expr::Int(2)),
                ),
            ),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(20));
    }

    #[test]
    fn test_vm_chained_arithmetic() {
        let e = Expr::binary(
            BinOp::Mul,
            Expr::binary(BinOp::Add, Expr::Int(1), Expr::Int(2)),
            Expr::binary(BinOp::Add, Expr::Int(3), Expr::Int(4)),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(21));
    }

    #[test]
    fn test_vm_double_neg() {
        let e = Expr::unary(UnaryOp::Neg, Expr::unary(UnaryOp::Neg, Expr::Int(42)));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(42));
    }

    #[test]
    fn test_vm_double_not() {
        let e = Expr::unary(UnaryOp::Not, Expr::unary(UnaryOp::Not, Expr::Bool(true)));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    // === Optimization tests ===

    #[test]
    fn test_constant_folding_add() {
        let e = Expr::binary(BinOp::Add, Expr::Int(3), Expr::Int(4));
        let mut program = Codegen::new().compile(&e);
        constant_folding(&mut program);
        let has_seven = program.instructions.iter().any(|ir| {
            matches!(
                ir,
                Ir::LoadConst {
                    val: Value::Int(7),
                    ..
                }
            )
        });
        assert!(has_seven);
    }

    #[test]
    fn test_constant_folding_mul() {
        let e = Expr::binary(BinOp::Mul, Expr::Int(6), Expr::Int(7));
        let mut program = Codegen::new().compile(&e);
        constant_folding(&mut program);
        let has_42 = program.instructions.iter().any(|ir| {
            matches!(
                ir,
                Ir::LoadConst {
                    val: Value::Int(42),
                    ..
                }
            )
        });
        assert!(has_42);
    }

    #[test]
    fn test_constant_folding_chain() {
        let e = Expr::binary(
            BinOp::Mul,
            Expr::binary(BinOp::Add, Expr::Int(2), Expr::Int(3)),
            Expr::Int(4),
        );
        let mut program = Codegen::new().compile(&e);
        constant_folding(&mut program);
        let has_20 = program.instructions.iter().any(|ir| {
            matches!(
                ir,
                Ir::LoadConst {
                    val: Value::Int(20),
                    ..
                }
            )
        });
        assert!(has_20);
    }

    #[test]
    fn test_constant_folding_float() {
        let e = Expr::binary(BinOp::Add, Expr::Float(1.5), Expr::Float(2.5));
        let mut program = Codegen::new().compile(&e);
        constant_folding(&mut program);
        let has_four = program.instructions.iter().any(|ir| {
            matches!(ir, Ir::LoadConst { val: Value::Float(v), .. } if (*v - 4.0).abs() < f64::EPSILON)
        });
        assert!(has_four);
    }

    #[test]
    fn test_constant_folding_neg() {
        let e = Expr::unary(UnaryOp::Neg, Expr::Int(5));
        let mut program = Codegen::new().compile(&e);
        constant_folding(&mut program);
        let has_neg5 = program.instructions.iter().any(|ir| {
            matches!(
                ir,
                Ir::LoadConst {
                    val: Value::Int(-5),
                    ..
                }
            )
        });
        assert!(has_neg5);
    }

    #[test]
    fn test_constant_folding_not() {
        let e = Expr::unary(UnaryOp::Not, Expr::Bool(true));
        let mut program = Codegen::new().compile(&e);
        constant_folding(&mut program);
        let has_false = program.instructions.iter().any(|ir| {
            matches!(
                ir,
                Ir::LoadConst {
                    val: Value::Bool(false),
                    ..
                }
            )
        });
        assert!(has_false);
    }

    #[test]
    fn test_constant_propagation() {
        let e = Expr::let_bind("x", Expr::Int(42), Expr::Var("x".into()));
        let mut program = Codegen::new().compile(&e);
        constant_propagation(&mut program);
        let load_count = program
            .instructions
            .iter()
            .filter(|ir| {
                matches!(
                    ir,
                    Ir::LoadConst {
                        val: Value::Int(42),
                        ..
                    }
                )
            })
            .count();
        assert!(load_count >= 2);
    }

    #[test]
    fn test_dead_code_elimination() {
        let e = Expr::if_expr(Expr::Bool(true), Expr::Int(1), Expr::Int(0));
        let mut program = Codegen::new().compile(&e);
        let before = program.instructions.len();
        dead_code_elimination(&mut program);
        let after = program.instructions.len();
        assert!(after <= before);
    }

    #[test]
    fn test_peephole_self_copy() {
        let mut program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Int(1),
                },
                Ir::Copy { dst: 0, src: 0 },
                Ir::Return(0),
            ],
            register_count: 1,
        };
        peephole(&mut program);
        assert!(!program
            .instructions
            .iter()
            .any(|ir| matches!(ir, Ir::Copy { .. })));
    }

    #[test]
    fn test_optimize_all_passes() {
        let e = Expr::binary(
            BinOp::Mul,
            Expr::binary(BinOp::Add, Expr::Int(2), Expr::Int(3)),
            Expr::binary(BinOp::Add, Expr::Int(4), Expr::Int(6)),
        );
        let result = compile_optimize_and_run(&e).unwrap();
        assert_eq!(result, Value::Int(50));
    }

    #[test]
    fn test_optimize_preserves_correctness() {
        let e = Expr::let_bind(
            "x",
            Expr::Int(10),
            Expr::binary(BinOp::Add, Expr::Var("x".into()), Expr::Int(5)),
        );
        let unopt = compile_and_run(&e).unwrap();
        let opt = compile_optimize_and_run(&e).unwrap();
        assert_eq!(unopt, opt);
    }

    #[test]
    fn test_optimize_complex() {
        let e = Expr::let_bind(
            "a",
            Expr::binary(BinOp::Add, Expr::Int(3), Expr::Int(2)),
            Expr::let_bind(
                "b",
                Expr::binary(BinOp::Mul, Expr::Var("a".into()), Expr::Int(4)),
                Expr::binary(BinOp::Sub, Expr::Var("b".into()), Expr::Int(1)),
            ),
        );
        assert_eq!(compile_optimize_and_run(&e).unwrap(), Value::Int(19));
    }

    // === IrProgram methods ===

    #[test]
    fn test_active_instruction_count() {
        let program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Int(1),
                },
                Ir::Nop,
                Ir::Return(0),
            ],
            register_count: 1,
        };
        assert_eq!(program.active_instruction_count(), 2);
    }

    #[test]
    fn test_program_dump() {
        let program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Int(42),
                },
                Ir::Return(0),
            ],
            register_count: 1,
        };
        let dump = program.dump();
        assert!(dump.contains("r0 = 42"));
        assert!(dump.contains("return r0"));
    }

    // === VM Default ===

    #[test]
    fn test_vm_default() {
        let vm = Vm::default();
        assert!(vm.registers.is_empty());
    }

    // === Error cases ===

    #[test]
    fn test_vm_div_by_zero() {
        let e = Expr::binary(BinOp::Div, Expr::Int(1), Expr::Int(0));
        let result = compile_and_run(&e);
        assert!(result.is_err());
    }

    #[test]
    fn test_vm_mod_by_zero() {
        let e = Expr::binary(BinOp::Mod, Expr::Int(1), Expr::Int(0));
        let result = compile_and_run(&e);
        assert!(result.is_err());
    }

    #[test]
    fn test_vm_type_mismatch() {
        let program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Int(1),
                },
                Ir::LoadConst {
                    dst: 1,
                    val: Value::Bool(true),
                },
                Ir::BinOp {
                    op: BinOp::Add,
                    dst: 2,
                    lhs: 0,
                    rhs: 1,
                },
                Ir::Return(2),
            ],
            register_count: 3,
        };
        let mut vm = Vm::new();
        assert!(vm.execute(&program).is_err());
    }

    // === Additional arithmetic / comparison edge cases ===

    #[test]
    fn test_vm_float_comparison_lt() {
        let e = Expr::binary(BinOp::Lt, Expr::Float(1.0), Expr::Float(2.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_float_comparison_ge() {
        let e = Expr::binary(BinOp::Ge, Expr::Float(2.0), Expr::Float(2.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_float_mod() {
        let e = Expr::binary(BinOp::Mod, Expr::Float(7.0), Expr::Float(3.0));
        let result = compile_and_run(&e).unwrap();
        if let Value::Float(v) = result {
            assert!((v - 1.0).abs() < f64::EPSILON);
        } else {
            panic!("expected float");
        }
    }

    #[test]
    fn test_vm_bool_eq() {
        let e = Expr::binary(BinOp::Eq, Expr::Bool(true), Expr::Bool(true));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_bool_ne() {
        let e = Expr::binary(BinOp::Ne, Expr::Bool(true), Expr::Bool(false));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_float_eq() {
        let e = Expr::binary(BinOp::Eq, Expr::Float(1.0), Expr::Float(1.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_float_ne() {
        let e = Expr::binary(BinOp::Ne, Expr::Float(1.0), Expr::Float(2.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_float_le() {
        let e = Expr::binary(BinOp::Le, Expr::Float(1.0), Expr::Float(1.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_float_gt() {
        let e = Expr::binary(BinOp::Gt, Expr::Float(3.0), Expr::Float(2.0));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_and_both_true() {
        let e = Expr::binary(BinOp::And, Expr::Bool(true), Expr::Bool(true));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_or_both_false() {
        let e = Expr::binary(BinOp::Or, Expr::Bool(false), Expr::Bool(false));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(false));
    }

    // === Wrapping arithmetic ===

    #[test]
    fn test_vm_wrapping_add() {
        let e = Expr::binary(BinOp::Add, Expr::Int(i64::MAX), Expr::Int(1));
        let result = compile_and_run(&e).unwrap();
        assert_eq!(result, Value::Int(i64::MIN));
    }

    #[test]
    fn test_vm_wrapping_sub() {
        let e = Expr::binary(BinOp::Sub, Expr::Int(i64::MIN), Expr::Int(1));
        let result = compile_and_run(&e).unwrap();
        assert_eq!(result, Value::Int(i64::MAX));
    }

    #[test]
    fn test_vm_wrapping_mul() {
        let e = Expr::binary(BinOp::Mul, Expr::Int(i64::MAX), Expr::Int(2));
        let result = compile_and_run(&e).unwrap();
        assert_eq!(result, Value::Int(-2));
    }

    // === Deeply nested expressions ===

    #[test]
    fn test_vm_deep_nesting() {
        let mut e = Expr::Int(1);
        for i in 2..=5 {
            e = Expr::binary(BinOp::Add, e, Expr::Int(i));
        }
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(15));
    }

    #[test]
    fn test_vm_deeply_nested_let() {
        let e = Expr::let_bind(
            "a",
            Expr::Int(1),
            Expr::let_bind(
                "b",
                Expr::binary(BinOp::Add, Expr::Var("a".into()), Expr::Int(1)),
                Expr::let_bind(
                    "c",
                    Expr::binary(BinOp::Add, Expr::Var("b".into()), Expr::Int(1)),
                    Expr::let_bind(
                        "d",
                        Expr::binary(BinOp::Add, Expr::Var("c".into()), Expr::Int(1)),
                        Expr::Var("d".into()),
                    ),
                ),
            ),
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(4));
    }

    // === Block with let bindings ===

    #[test]
    fn test_vm_block_with_let() {
        let e = Expr::Block(vec![Expr::let_bind(
            "x",
            Expr::Int(10),
            Expr::binary(BinOp::Add, Expr::Var("x".into()), Expr::Int(5)),
        )]);
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(15));
    }

    // === Value display edge cases ===

    #[test]
    fn test_value_display_float() {
        let v = Value::Float(2.719);
        let s = format!("{v}");
        assert!(s.starts_with("2.719"));
    }

    #[test]
    fn test_value_display_negative_int() {
        let v = Value::Int(-42);
        assert_eq!(format!("{v}"), "-42");
    }

    // === Compile optimize and run ===

    #[test]
    fn test_compile_optimize_simple() {
        let e = Expr::Int(42);
        assert_eq!(compile_optimize_and_run(&e).unwrap(), Value::Int(42));
    }

    #[test]
    fn test_compile_optimize_if() {
        let e = Expr::if_expr(
            Expr::binary(BinOp::Eq, Expr::Int(1), Expr::Int(1)),
            Expr::Int(100),
            Expr::Int(200),
        );
        assert_eq!(compile_optimize_and_run(&e).unwrap(), Value::Int(100));
    }

    #[test]
    fn test_compile_optimize_nested_arithmetic() {
        let e = Expr::binary(
            BinOp::Sub,
            Expr::binary(
                BinOp::Mul,
                Expr::binary(BinOp::Add, Expr::Int(1), Expr::Int(2)),
                Expr::Int(3),
            ),
            Expr::binary(
                BinOp::Mul,
                Expr::binary(BinOp::Add, Expr::Int(4), Expr::Int(5)),
                Expr::Int(2),
            ),
        );
        assert_eq!(compile_optimize_and_run(&e).unwrap(), Value::Int(-9));
    }

    // === Fn / Call ===

    #[test]
    fn test_vm_fn_inline() {
        let e = Expr::Fn {
            name: None,
            params: vec![],
            body: Box::new(Expr::Int(99)),
        };
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(99));
    }

    #[test]
    fn test_vm_call_inline() {
        let e = Expr::call(
            Expr::Fn {
                name: None,
                params: vec![],
                body: Box::new(Expr::Int(77)),
            },
            vec![],
        );
        assert_eq!(compile_and_run(&e).unwrap(), Value::Int(77));
    }

    // === Constant folding string concat ===

    #[test]
    fn test_constant_folding_string() {
        let e = Expr::binary(BinOp::Add, Expr::Str("foo".into()), Expr::Str("bar".into()));
        let mut program = Codegen::new().compile(&e);
        constant_folding(&mut program);
        let has_foobar = program
            .instructions
            .iter()
            .any(|ir| matches!(ir, Ir::LoadConst { val: Value::Str(s), .. } if s == "foobar"));
        assert!(has_foobar);
    }

    // === JumpIfTrue in VM ===

    #[test]
    fn test_vm_jump_if_true() {
        let program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Bool(true),
                },
                Ir::LoadConst {
                    dst: 1,
                    val: Value::Int(10),
                },
                Ir::JumpIfTrue { cond: 0, target: 0 },
                Ir::LoadConst {
                    dst: 1,
                    val: Value::Int(20),
                },
                Ir::Label(0),
                Ir::Return(1),
            ],
            register_count: 2,
        };
        let mut vm = Vm::new();
        assert_eq!(vm.execute(&program).unwrap(), Value::Int(10));
    }

    #[test]
    fn test_vm_jump_if_true_not_taken() {
        let program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Bool(false),
                },
                Ir::LoadConst {
                    dst: 1,
                    val: Value::Int(10),
                },
                Ir::JumpIfTrue { cond: 0, target: 0 },
                Ir::LoadConst {
                    dst: 1,
                    val: Value::Int(20),
                },
                Ir::Label(0),
                Ir::Return(1),
            ],
            register_count: 2,
        };
        let mut vm = Vm::new();
        assert_eq!(vm.execute(&program).unwrap(), Value::Int(20));
    }

    // === Edge: empty program returns Unit ===

    #[test]
    fn test_vm_no_return() {
        let program = IrProgram {
            instructions: vec![Ir::LoadConst {
                dst: 0,
                val: Value::Int(1),
            }],
            register_count: 1,
        };
        let mut vm = Vm::new();
        assert_eq!(vm.execute(&program).unwrap(), Value::Unit);
    }

    // === Unary on typed error ===

    #[test]
    fn test_vm_neg_bool_error() {
        let program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Bool(true),
                },
                Ir::UnaryOp {
                    op: UnaryOp::Neg,
                    dst: 1,
                    src: 0,
                },
                Ir::Return(1),
            ],
            register_count: 2,
        };
        let mut vm = Vm::new();
        assert!(vm.execute(&program).is_err());
    }

    #[test]
    fn test_vm_not_int_error() {
        let program = IrProgram {
            instructions: vec![
                Ir::LoadConst {
                    dst: 0,
                    val: Value::Int(5),
                },
                Ir::UnaryOp {
                    op: UnaryOp::Not,
                    dst: 1,
                    src: 0,
                },
                Ir::Return(1),
            ],
            register_count: 2,
        };
        let mut vm = Vm::new();
        assert!(vm.execute(&program).is_err());
    }

    // === Int comparison edge cases ===

    #[test]
    fn test_vm_int_le_not_equal() {
        let e = Expr::binary(BinOp::Le, Expr::Int(3), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_int_gt_false() {
        let e = Expr::binary(BinOp::Gt, Expr::Int(3), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_vm_int_ge_equal() {
        let e = Expr::binary(BinOp::Ge, Expr::Int(5), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_vm_int_ne_same() {
        let e = Expr::binary(BinOp::Ne, Expr::Int(5), Expr::Int(5));
        assert_eq!(compile_and_run(&e).unwrap(), Value::Bool(false));
    }
}
