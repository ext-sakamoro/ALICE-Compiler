//! Fuzz target: 攻撃者制御 byte から AST expression を組み立て、compile → run が panic しないことを検証
//!
//! 攻撃者制御 (op tag / literal / nesting depth) の AST 構築で:
//! - 深い recursion による stack overflow
//! - Div/Mod by zero (compile_and_run は Err で返すべき、panic は不許可)
//! - 型不整合 (String + Int 等) の handling
//! - Box::new 大量生成による allocation panic
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠
//! ast/ir types が Arbitrary を derive していないため Unstructured から手動構築

#![no_main]

use alice_compiler::ast::{BinOp, Expr, UnaryOp};
use alice_compiler::compile::compile_and_run;
use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

const MAX_DEPTH: u8 = 6;
const MAX_ARGS: usize = 4;
const MAX_BLOCK_LEN: usize = 4;
const MAX_STR_LEN: usize = 16;

fn arb_binop(u: &mut Unstructured<'_>) -> arbitrary::Result<BinOp> {
    Ok(match u.int_in_range::<u8>(0..=12)? {
        0 => BinOp::Add,
        1 => BinOp::Sub,
        2 => BinOp::Mul,
        3 => BinOp::Div,
        4 => BinOp::Mod,
        5 => BinOp::Eq,
        6 => BinOp::Ne,
        7 => BinOp::Lt,
        8 => BinOp::Le,
        9 => BinOp::Gt,
        10 => BinOp::Ge,
        11 => BinOp::And,
        _ => BinOp::Or,
    })
}

fn arb_unop(u: &mut Unstructured<'_>) -> arbitrary::Result<UnaryOp> {
    Ok(if u.arbitrary::<bool>()? { UnaryOp::Neg } else { UnaryOp::Not })
}

fn arb_short_str(u: &mut Unstructured<'_>) -> arbitrary::Result<String> {
    let len = u.int_in_range::<usize>(0..=MAX_STR_LEN)?;
    let bytes = u.bytes(len)?;
    // 制御文字を除去して簡易 identifier / string literal 化
    Ok(bytes.iter().map(|&b| ((b % 26) + b'a') as char).collect())
}

fn arb_expr(u: &mut Unstructured<'_>, depth: u8) -> arbitrary::Result<Expr> {
    // depth 上限で literal に強制降格 (stack overflow 防止)
    if depth >= MAX_DEPTH || u.is_empty() {
        return Ok(match u.int_in_range::<u8>(0..=4)? {
            0 => Expr::Int(u.arbitrary::<i64>()?),
            1 => Expr::Float(u.arbitrary::<f64>()?),
            2 => Expr::Bool(u.arbitrary::<bool>()?),
            3 => Expr::Str(arb_short_str(u)?),
            _ => Expr::Var(arb_short_str(u)?),
        });
    }
    Ok(match u.int_in_range::<u8>(0..=11)? {
        0 => Expr::Int(u.arbitrary::<i64>()?),
        1 => Expr::Float(u.arbitrary::<f64>()?),
        2 => Expr::Bool(u.arbitrary::<bool>()?),
        3 => Expr::Str(arb_short_str(u)?),
        4 => Expr::Var(arb_short_str(u)?),
        5 => Expr::binary(arb_binop(u)?, arb_expr(u, depth + 1)?, arb_expr(u, depth + 1)?),
        6 => Expr::unary(arb_unop(u)?, arb_expr(u, depth + 1)?),
        7 => Expr::let_bind(arb_short_str(u)?, arb_expr(u, depth + 1)?, arb_expr(u, depth + 1)?),
        8 => Expr::if_expr(
            arb_expr(u, depth + 1)?,
            arb_expr(u, depth + 1)?,
            arb_expr(u, depth + 1)?,
        ),
        9 => {
            let n = u.int_in_range::<usize>(0..=MAX_ARGS)?;
            let mut params = Vec::with_capacity(n);
            for _ in 0..n {
                params.push(arb_short_str(u)?);
            }
            let name = if u.arbitrary::<bool>()? { Some(arb_short_str(u)?) } else { None };
            Expr::Fn { name, params, body: Box::new(arb_expr(u, depth + 1)?) }
        }
        10 => {
            let n = u.int_in_range::<usize>(0..=MAX_ARGS)?;
            let mut args = Vec::with_capacity(n);
            for _ in 0..n {
                args.push(arb_expr(u, depth + 1)?);
            }
            Expr::call(arb_expr(u, depth + 1)?, args)
        }
        _ => {
            let n = u.int_in_range::<usize>(0..=MAX_BLOCK_LEN)?;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                items.push(arb_expr(u, depth + 1)?);
            }
            Expr::Block(items)
        }
    })
}

fuzz_target!(|data: &[u8]| {
    // 巨大 input による fuzzer timeout 回避
    if data.len() > 64 * 1024 {
        return;
    }
    let mut u = Unstructured::new(data);
    let Ok(expr) = arb_expr(&mut u, 0) else {
        return;
    };
    // Result は誤入力で Err を返してよい (panic ゼロが gate)
    let _ = compile_and_run(&expr);
});
