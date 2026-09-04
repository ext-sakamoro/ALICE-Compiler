//! Fuzz target: 攻撃者制御 IrProgram に optimizer (constant folding / propagation / DCE / peephole) を適用しても panic しないことを検証
//!
//! 攻撃者制御 IR で:
//! - constant_folding の Div/Mod by zero 分岐 → panic 不許可
//! - dead_code_elimination の label reachability 走査 → OOB や cycle で panic 不許可
//! - peephole の pattern match 走査 → OOB 不許可
//! - constant_propagation の register 書き換え → OOB 不許可
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use alice_compiler::ast::{BinOp, UnaryOp};
use alice_compiler::ir::{Ir, IrProgram, LabelId, Reg, Value};
use alice_compiler::optimization::{
    constant_folding, constant_propagation, dead_code_elimination, optimize, peephole,
};
use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

const MAX_INSTRUCTIONS: usize = 128;
const MAX_REGISTERS: usize = 32;
const MAX_LABELS: usize = 16;
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

fn arb_value(u: &mut Unstructured<'_>) -> arbitrary::Result<Value> {
    Ok(match u.int_in_range::<u8>(0..=4)? {
        0 => Value::Int(u.arbitrary::<i64>()?),
        1 => Value::Float(u.arbitrary::<f64>()?),
        2 => Value::Bool(u.arbitrary::<bool>()?),
        3 => {
            let len = u.int_in_range::<usize>(0..=MAX_STR_LEN)?;
            let bytes = u.bytes(len)?;
            let s: String = bytes.iter().map(|&b| ((b % 26) + b'a') as char).collect();
            Value::Str(s)
        }
        _ => Value::Unit,
    })
}

fn arb_reg(u: &mut Unstructured<'_>, register_count: usize) -> arbitrary::Result<Reg> {
    if register_count == 0 {
        return Ok(0);
    }
    Ok(u.int_in_range::<usize>(0..=register_count - 1)?)
}

fn arb_label(u: &mut Unstructured<'_>) -> arbitrary::Result<LabelId> {
    Ok(u.int_in_range::<usize>(0..=MAX_LABELS.saturating_sub(1))?)
}

fn arb_program(u: &mut Unstructured<'_>) -> Option<IrProgram> {
    let register_count = u.int_in_range::<usize>(1..=MAX_REGISTERS).ok()?;
    let n = u.int_in_range::<usize>(0..=MAX_INSTRUCTIONS).ok()?;
    let mut instructions: Vec<Ir> = Vec::with_capacity(n);
    for _ in 0..n {
        let Ok(tag) = u.int_in_range::<u8>(0..=9) else { break };
        let ir = match tag {
            0 => Ir::LoadConst {
                dst: arb_reg(u, register_count).ok()?,
                val: arb_value(u).ok()?,
            },
            1 => Ir::Copy {
                dst: arb_reg(u, register_count).ok()?,
                src: arb_reg(u, register_count).ok()?,
            },
            2 => Ir::BinOp {
                op: arb_binop(u).ok()?,
                dst: arb_reg(u, register_count).ok()?,
                lhs: arb_reg(u, register_count).ok()?,
                rhs: arb_reg(u, register_count).ok()?,
            },
            3 => Ir::UnaryOp {
                op: arb_unop(u).ok()?,
                dst: arb_reg(u, register_count).ok()?,
                src: arb_reg(u, register_count).ok()?,
            },
            4 => Ir::Label(arb_label(u).ok()?),
            5 => Ir::Jump(arb_label(u).ok()?),
            6 => Ir::JumpIfFalse {
                cond: arb_reg(u, register_count).ok()?,
                target: arb_label(u).ok()?,
            },
            7 => Ir::JumpIfTrue {
                cond: arb_reg(u, register_count).ok()?,
                target: arb_label(u).ok()?,
            },
            8 => Ir::Return(arb_reg(u, register_count).ok()?),
            _ => Ir::Nop,
        };
        instructions.push(ir);
    }
    Some(IrProgram { instructions, register_count })
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    let mut u = Unstructured::new(data);
    let Some(mut program) = arb_program(&mut u) else {
        return;
    };

    // 個別 pass も panic ゼロで完了すること (順序独立で robust であるべき)
    let mut clone_a = program.clone();
    constant_folding(&mut clone_a);

    let mut clone_b = program.clone();
    constant_propagation(&mut clone_b);

    let mut clone_c = program.clone();
    dead_code_elimination(&mut clone_c);

    let mut clone_d = program.clone();
    peephole(&mut clone_d);

    // フル optimize (canonical 順で 4 pass 適用)
    optimize(&mut program);
});
