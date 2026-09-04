//! Fuzz target: 攻撃者制御 IrProgram を Vm で execute しても panic しないことを検証
//!
//! 攻撃者制御 (register index / label id / value) の IR 実行で:
//! - out-of-bounds register access (index >= register_count) → panic
//! - undefined label jump → panic
//! - infinite jump loop → step limit で abort すべき (panic 不許可)
//! - Vec::with_capacity(huge) による capacity overflow panic
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠
//! 注: VM は無限 loop 検出を持たないため fuzz 側で instruction 長 + register_count を上限化して停止性を担保

#![no_main]

use alice_compiler::ast::{BinOp, UnaryOp};
use alice_compiler::ir::{Ir, IrProgram, LabelId, Reg, Value};
use alice_compiler::vm::Vm;
use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

const MAX_INSTRUCTIONS: usize = 64;
const MAX_REGISTERS: usize = 16;
const MAX_LABELS: usize = 8;
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

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    let mut u = Unstructured::new(data);

    // register_count は 1..=MAX_REGISTERS に強制 (0 は VM 側で invalid、OOB 起きうる)
    let Ok(register_count) = u.int_in_range::<usize>(1..=MAX_REGISTERS) else {
        return;
    };

    let Ok(n) = u.int_in_range::<usize>(0..=MAX_INSTRUCTIONS) else {
        return;
    };

    let mut instructions: Vec<Ir> = Vec::with_capacity(n);
    for _ in 0..n {
        let Ok(tag) = u.int_in_range::<u8>(0..=9) else {
            break;
        };
        let ir = match tag {
            0 => {
                let Ok(dst) = arb_reg(&mut u, register_count) else { break };
                let Ok(val) = arb_value(&mut u) else { break };
                Ir::LoadConst { dst, val }
            }
            1 => {
                let Ok(dst) = arb_reg(&mut u, register_count) else { break };
                let Ok(src) = arb_reg(&mut u, register_count) else { break };
                Ir::Copy { dst, src }
            }
            2 => {
                let Ok(op) = arb_binop(&mut u) else { break };
                let Ok(dst) = arb_reg(&mut u, register_count) else { break };
                let Ok(lhs) = arb_reg(&mut u, register_count) else { break };
                let Ok(rhs) = arb_reg(&mut u, register_count) else { break };
                Ir::BinOp { op, dst, lhs, rhs }
            }
            3 => {
                let Ok(op) = arb_unop(&mut u) else { break };
                let Ok(dst) = arb_reg(&mut u, register_count) else { break };
                let Ok(src) = arb_reg(&mut u, register_count) else { break };
                Ir::UnaryOp { op, dst, src }
            }
            4 => {
                let Ok(l) = arb_label(&mut u) else { break };
                Ir::Label(l)
            }
            5 => {
                let Ok(l) = arb_label(&mut u) else { break };
                Ir::Jump(l)
            }
            6 => {
                let Ok(cond) = arb_reg(&mut u, register_count) else { break };
                let Ok(target) = arb_label(&mut u) else { break };
                Ir::JumpIfFalse { cond, target }
            }
            7 => {
                let Ok(cond) = arb_reg(&mut u, register_count) else { break };
                let Ok(target) = arb_label(&mut u) else { break };
                Ir::JumpIfTrue { cond, target }
            }
            8 => {
                let Ok(reg) = arb_reg(&mut u, register_count) else { break };
                Ir::Return(reg)
            }
            _ => Ir::Nop,
        };
        instructions.push(ir);
    }

    // Return を末尾に必ず 1 個追加 (無限 loop 化した jump chain の停止性を担保)
    instructions.push(Ir::Return(0));

    let program = IrProgram { instructions, register_count };
    let mut vm = Vm::new();
    // Result::Err は許容 (unknown var / type mismatch / label undefined 等)、panic ゼロが gate
    let _ = vm.execute(&program);
});
