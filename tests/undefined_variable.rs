//! An undefined variable is an error of compilation, not a panic.
//!
//! oracle: a name is defined inside the body of the `let` that binds it and nowhere else. The AST is
//! input (`compile_and_run` already returns `Err` for what fails at run time), so a name with no
//! binding gives `CompileError::UndefinedVariable` naming it, from every entry point.

use alice_compiler::ast::{BinOp, Expr};
use alice_compiler::codegen::{Codegen, CompileError};
use alice_compiler::compile::{compile_and_run, compile_optimize_and_run};
use alice_compiler::ir::Value;

fn var(name: &str) -> Expr {
    Expr::Var(name.to_owned())
}

fn boxed(e: Expr) -> Box<Expr> {
    Box::new(e)
}

fn undefined(name: &str) -> CompileError {
    CompileError::UndefinedVariable {
        name: name.to_owned(),
    }
}

#[test]
fn every_place_a_name_can_be() {
    let name = "x";
    for e in [
        var(name),
        Expr::Binary {
            op: BinOp::Add,
            lhs: boxed(Expr::Int(1)),
            rhs: boxed(var(name)),
        },
        Expr::If {
            cond: boxed(var(name)),
            then: boxed(Expr::Int(1)),
            else_: boxed(Expr::Int(2)),
        },
        Expr::If {
            cond: boxed(Expr::Bool(true)),
            then: boxed(Expr::Int(1)),
            else_: boxed(var(name)),
        },
        Expr::Block(vec![Expr::Int(1), var(name)]),
        Expr::Let {
            name: "y".to_owned(),
            value: boxed(var(name)),
            body: boxed(Expr::Int(1)),
        },
    ] {
        assert_eq!(
            Codegen::new().compile(&e).map(|_| ()),
            Err(undefined(name)),
            "{e:?}"
        );
        assert!(
            compile_and_run(&e).is_err_and(|m| m.contains(name)),
            "{e:?}"
        );
        assert!(compile_optimize_and_run(&e).is_err(), "{e:?}");
    }
}

#[test]
fn the_scope_of_a_let_ends_with_its_body() {
    let e = Expr::Block(vec![
        Expr::Let {
            name: "x".to_owned(),
            value: boxed(Expr::Int(1)),
            body: boxed(var("x")),
        },
        var("x"),
    ]);
    assert_eq!(Codegen::new().compile(&e).map(|_| ()), Err(undefined("x")));
}

#[test]
fn a_bound_name_still_compiles() {
    let e = Expr::Let {
        name: "x".to_owned(),
        value: boxed(Expr::Int(5)),
        body: boxed(var("x")),
    };
    assert_eq!(compile_and_run(&e), Ok(Value::Int(5)));
    assert_eq!(compile_optimize_and_run(&e), Ok(Value::Int(5)));
    // the empty name of the fuzz crash is a name like another
    assert_eq!(
        Codegen::new().compile(&var("")).map(|_| ()),
        Err(undefined(""))
    );
}
