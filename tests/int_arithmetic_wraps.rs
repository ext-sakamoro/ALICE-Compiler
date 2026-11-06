//! Integer negation and division have a defined result at the edge of `i64`, in both profiles.
//!
//! oracle: `Value::Int` arithmetic is two's complement with wrapping, as `+`, `-` and `*` already
//! are: `-x` is `0 - x`, and at the one quotient `i64` cannot hold (`MIN / -1`) `a / b` is `-a` and
//! `a % b` is 0, which is what `a == (a / b) * b + a % b` asks of them. Evaluated by the VM and by
//! the optimizer's constant folding, which must agree. Without overflow checks these already
//! wrapped; with them `-MIN` (the fuzz crash), `MIN / -1` and `MIN % -1` panicked.

use alice_compiler::ast::{BinOp, UnaryOp};
use alice_compiler::ir::{Ir, IrProgram, Value};
use alice_compiler::optimization::optimize;
use alice_compiler::vm::Vm;

const MIN: i64 = i64::MIN;

fn run(instructions: Vec<Ir>, register_count: usize) -> Result<Value, String> {
    Vm::new().execute(&IrProgram {
        instructions,
        register_count,
    })
}

fn neg(v: i64) -> Vec<Ir> {
    vec![
        Ir::LoadConst {
            dst: 0,
            val: Value::Int(v),
        },
        Ir::UnaryOp {
            op: UnaryOp::Neg,
            dst: 1,
            src: 0,
        },
        Ir::Return(1),
    ]
}

fn binary(op: BinOp, a: i64, b: i64) -> Vec<Ir> {
    vec![
        Ir::LoadConst {
            dst: 0,
            val: Value::Int(a),
        },
        Ir::LoadConst {
            dst: 1,
            val: Value::Int(b),
        },
        Ir::BinOp {
            op,
            dst: 2,
            lhs: 0,
            rhs: 1,
        },
        Ir::Return(2),
    ]
}

/// The result both as written and after the optimizer's constant folding.
fn both(
    instructions: Vec<Ir>,
    register_count: usize,
) -> (Result<Value, String>, Result<Value, String>) {
    let plain = run(instructions.clone(), register_count);
    let mut program = IrProgram {
        instructions,
        register_count,
    };
    optimize(&mut program);
    (plain, Vm::new().execute(&program))
}

#[test]
fn negation() {
    for (v, expected) in [
        (MIN, MIN),
        (MIN + 1, i64::MAX),
        (i64::MAX, MIN + 1),
        (-1, 1),
        (0, 0),
        (5, -5),
    ] {
        let (plain, folded) = both(neg(v), 2);
        assert_eq!(plain, Ok(Value::Int(expected)), "-({v})");
        assert_eq!(folded, Ok(Value::Int(expected)), "-({v}), folded");
    }
}

#[test]
fn negation_is_zero_minus_x() {
    for v in [MIN, MIN + 1, -7, 0, 7, i64::MAX] {
        assert_eq!(
            run(neg(v), 2),
            run(binary(BinOp::Sub, 0, v), 3),
            "-({v}) and 0 - {v}"
        );
    }
}

#[test]
fn division_at_the_edge() {
    for (op, a, b, expected) in [
        (BinOp::Div, MIN, -1, MIN),
        (BinOp::Mod, MIN, -1, 0),
        (BinOp::Div, MIN, 1, MIN),
        (BinOp::Mod, MIN, 1, 0),
        (BinOp::Div, MIN + 1, -1, i64::MAX),
        (BinOp::Div, 7, -2, -3),
        (BinOp::Mod, 7, -2, 1),
        (BinOp::Div, -7, 2, -3),
        (BinOp::Mod, -7, 2, -1),
    ] {
        let (plain, folded) = both(binary(op, a, b), 3);
        assert_eq!(plain, Ok(Value::Int(expected)), "{a} {op} {b}");
        assert_eq!(folded, Ok(Value::Int(expected)), "{a} {op} {b}, folded");
    }
}

#[test]
fn division_by_zero_stays_an_error() {
    for op in [BinOp::Div, BinOp::Mod] {
        for a in [MIN, 0, 5] {
            assert!(run(binary(op, a, 0), 3).is_err(), "{a} {op} 0");
        }
    }
}
