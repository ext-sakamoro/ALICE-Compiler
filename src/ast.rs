//! Abstract syntax tree.

use std::fmt;

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
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Var(String),
    Binary {
        op: BinOp,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Self>,
    },
    Let {
        name: String,
        value: Box<Self>,
        body: Box<Self>,
    },
    If {
        cond: Box<Self>,
        then: Box<Self>,
        else_: Box<Self>,
    },
    Fn {
        name: Option<String>,
        params: Vec<String>,
        body: Box<Self>,
    },
    Call {
        func: Box<Self>,
        args: Vec<Self>,
    },
    Block(Vec<Self>),
}

impl Expr {
    #[must_use]
    pub fn binary(op: BinOp, lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    #[must_use]
    pub fn unary(op: UnaryOp, operand: Self) -> Self {
        Self::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    #[must_use]
    pub fn let_bind(name: impl Into<String>, value: Self, body: Self) -> Self {
        Self::Let {
            name: name.into(),
            value: Box::new(value),
            body: Box::new(body),
        }
    }

    #[must_use]
    pub fn if_expr(cond: Self, then: Self, else_: Self) -> Self {
        Self::If {
            cond: Box::new(cond),
            then: Box::new(then),
            else_: Box::new(else_),
        }
    }

    #[must_use]
    pub fn call(func: Self, args: Vec<Self>) -> Self {
        Self::Call {
            func: Box::new(func),
            args,
        }
    }

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
