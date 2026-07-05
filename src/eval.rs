//! Shared evaluation helpers for constant folding + VM.

use crate::ast::{BinOp, UnaryOp};
use crate::ir::Value;

pub fn eval_binop(op: BinOp, lhs: &Value, rhs: &Value) -> Option<Value> {
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

pub const fn eval_unaryop(op: UnaryOp, val: &Value) -> Option<Value> {
    match (op, val) {
        (UnaryOp::Neg, Value::Int(v)) => Some(Value::Int(-*v)),
        (UnaryOp::Neg, Value::Float(v)) => Some(Value::Float(-*v)),
        (UnaryOp::Not, Value::Bool(v)) => Some(Value::Bool(!*v)),
        _ => None,
    }
}
