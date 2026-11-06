//! `Vm::execute` returns `Err` for an `IrProgram` it cannot run, whatever the program holds.
//!
//! oracle: an `IrProgram` is input: `register_count` and every `Reg` and `LabelId` are plain numbers
//! the caller gives. A register below `register_count` and a label that exists are the only ones
//! the program defines, so anything else is an error, found before the first instruction runs: a
//! bad jump on a path not taken this time is still a bad program.

use alice_compiler::ast::{BinOp, UnaryOp};
use alice_compiler::ir::{Ir, IrProgram, Value};
use alice_compiler::vm::Vm;

fn run(instructions: Vec<Ir>, register_count: usize) -> Result<Value, String> {
    Vm::new().execute(&IrProgram {
        instructions,
        register_count,
    })
}

fn load(dst: usize) -> Ir {
    Ir::LoadConst {
        dst,
        val: Value::Bool(true),
    }
}

#[test]
fn an_undefined_label() {
    for jump in [
        Ir::Jump(9),
        Ir::JumpIfFalse { cond: 0, target: 9 },
        Ir::JumpIfTrue { cond: 0, target: 9 },
    ] {
        let err = run(vec![load(0), jump.clone(), Ir::Return(0)], 1);
        assert!(
            err.as_ref().is_err_and(|e| e.contains('9')),
            "{jump:?}: {err:?}"
        );
    }
}

#[test]
fn an_undefined_label_on_a_path_not_taken() {
    // the condition is true, so this jump is never taken
    let program = vec![
        load(0),
        Ir::JumpIfFalse { cond: 0, target: 9 },
        Ir::Return(0),
    ];
    assert!(run(program, 1).is_err());
    // nor is this one reached
    assert!(run(vec![load(0), Ir::Return(0), Ir::Jump(9)], 1).is_err());
}

#[test]
fn the_crash_input_of_the_fuzz_run_has_this_form() {
    // fuzz_ir_execute panicked at `label_map[target]`: a jump with no such label in a short program
    assert!(run(vec![Ir::Jump(0)], 1).is_err());
    assert!(run(vec![Ir::Jump(0)], 0).is_err());
}

#[test]
fn a_defined_label_still_jumps() {
    let program = vec![
        Ir::LoadConst {
            dst: 0,
            val: Value::Int(1),
        },
        Ir::Jump(7),
        Ir::LoadConst {
            dst: 0,
            val: Value::Int(2),
        },
        Ir::Label(7),
        Ir::Return(0),
    ];
    assert_eq!(run(program, 1), Ok(Value::Int(1)));
}

#[test]
fn a_register_outside_the_program() {
    let bad = 4;
    for ir in [
        load(bad),
        Ir::Copy { dst: bad, src: 0 },
        Ir::Copy { dst: 0, src: bad },
        Ir::BinOp {
            op: BinOp::Add,
            dst: bad,
            lhs: 0,
            rhs: 0,
        },
        Ir::BinOp {
            op: BinOp::Add,
            dst: 0,
            lhs: bad,
            rhs: 0,
        },
        Ir::BinOp {
            op: BinOp::Add,
            dst: 0,
            lhs: 0,
            rhs: bad,
        },
        Ir::UnaryOp {
            op: UnaryOp::Neg,
            dst: bad,
            src: 0,
        },
        Ir::UnaryOp {
            op: UnaryOp::Neg,
            dst: 0,
            src: bad,
        },
        Ir::Return(bad),
    ] {
        let got = run(vec![load(0), ir.clone()], 4);
        assert!(
            got.as_ref().is_err_and(|e| e.contains('4')),
            "{ir:?}: {got:?}"
        );
    }
    for ir in [
        Ir::JumpIfFalse {
            cond: bad,
            target: 0,
        },
        Ir::JumpIfTrue {
            cond: bad,
            target: 0,
        },
    ] {
        assert!(run(vec![Ir::Label(0), ir.clone()], 4).is_err(), "{ir:?}");
    }
    // the last register is inside
    assert_eq!(run(vec![load(3), Ir::Return(3)], 4), Ok(Value::Bool(true)));
    // no registers: every use is outside
    assert!(run(vec![Ir::Return(0)], 0).is_err());
}

#[test]
fn a_register_count_too_large_to_allocate() {
    assert!(run(vec![load(0), Ir::Return(0)], usize::MAX).is_err());
    assert!(run(vec![], usize::MAX).is_err());
}

#[test]
fn an_empty_program_returns_unit() {
    assert_eq!(run(vec![], 0), Ok(Value::Unit));
    assert_eq!(run(vec![Ir::Nop, Ir::Label(1)], 0), Ok(Value::Unit));
}
