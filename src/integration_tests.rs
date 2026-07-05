//! Integration tests moved from monolithic lib.rs.

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::redundant_clone,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::manual_midpoint
)]

use crate::ast::{BinOp, Expr, UnaryOp};
use crate::codegen::Codegen;
use crate::compile::{compile_and_run, compile_optimize_and_run};
use crate::ir::{Ir, IrProgram, Value};
use crate::optimization::{
    constant_folding, constant_propagation, dead_code_elimination, peephole,
};
use crate::vm::Vm;

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
