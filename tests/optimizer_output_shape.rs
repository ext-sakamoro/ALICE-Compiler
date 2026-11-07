//! What `optimize` leaves behind: each pass it runs has a program whose optimized form only that
//! pass produces.
//!
//! A pass that is skipped changes no result, so comparing results (`tests/optimizer_keeps_meaning.rs`)
//! cannot see it. These rows compare the instructions, written out by hand.

use alice_compiler::ast::BinOp;
use alice_compiler::ir::{Ir, IrProgram, Value};
use alice_compiler::optimization::optimize;

fn load(dst: usize, v: i64) -> Ir {
    Ir::LoadConst {
        dst,
        val: Value::Int(v),
    }
}

fn optimized(instructions: Vec<Ir>, register_count: usize) -> Vec<Ir> {
    let mut p = IrProgram {
        instructions,
        register_count,
    };
    optimize(&mut p);
    p.instructions
}

#[test]
fn a_copy_of_a_constant_becomes_the_constant() {
    let out = optimized(
        vec![load(0, 7), Ir::Copy { dst: 1, src: 0 }, Ir::Return(1)],
        2,
    );
    assert_eq!(out, vec![load(0, 7), load(1, 7), Ir::Return(1)]);
}

#[test]
fn an_operation_on_constants_becomes_its_value() {
    let out = optimized(
        vec![
            load(0, 2),
            load(1, 3),
            Ir::BinOp {
                op: BinOp::Add,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            Ir::Return(2),
        ],
        3,
    );
    assert_eq!(out, vec![load(0, 2), load(1, 3), load(2, 5), Ir::Return(2)]);
}

#[test]
fn code_after_a_jump_goes_up_to_the_next_label() {
    let out = optimized(
        vec![
            Ir::Jump(0),
            load(0, 1),
            Ir::Label(0),
            load(0, 2),
            Ir::Return(0),
        ],
        1,
    );
    assert_eq!(
        out,
        vec![Ir::Jump(0), Ir::Label(0), load(0, 2), Ir::Return(0)]
    );
}

#[test]
fn a_copy_to_itself_and_a_nop_go() {
    let out = optimized(
        vec![
            Ir::JumpIfFalse { cond: 0, target: 0 },
            Ir::Label(0),
            Ir::Copy { dst: 0, src: 0 },
            Ir::Nop,
            Ir::Return(0),
        ],
        1,
    );
    assert_eq!(
        out,
        vec![
            Ir::JumpIfFalse { cond: 0, target: 0 },
            Ir::Label(0),
            Ir::Return(0)
        ]
    );
}
