//! `optimize` keeps what a program means: the optimized program returns the same value or the same
//! refusal as the original.
//!
//! oracle: the unoptimized VM is the definition of what a program means, and each generated program
//! runs both ways. The rows are programs with a known answer from reading them by hand. A program
//! that runs into a limit is not compared, since optimizing changes how many steps and bytes it takes;
//! the count of programs compared is asserted, so a generator that only hits limits fails.

use alice_compiler::ast::{BinOp, UnaryOp};
use alice_compiler::ir::{Ir, IrProgram, Value};
use alice_compiler::optimization::{
    constant_folding, constant_propagation, dead_code_elimination, optimize, peephole,
};
use alice_compiler::vm::{Vm, VmLimits};

const LIMITS: VmLimits = VmLimits {
    max_steps: 2_000,
    max_string_bytes: 1 << 16,
    max_registers: 64,
};

fn run(p: &IrProgram) -> Result<Value, String> {
    Vm::with_limits(LIMITS).execute(p)
}

fn is_limit(r: &Result<Value, String>) -> bool {
    r.as_ref().is_err_and(|e| e.contains("over the limit"))
}

/// The same value (a `NaN` is the same as a `NaN`), or both refused.
fn same(a: &Result<Value, String>, b: &Result<Value, String>) -> bool {
    match (a, b) {
        (Ok(Value::Float(x)), Ok(Value::Float(y))) => x.to_bits() == y.to_bits(),
        (Ok(x), Ok(y)) => x == y,
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

fn int(dst: usize, v: i64) -> Ir {
    Ir::LoadConst {
        dst,
        val: Value::Int(v),
    }
}

fn add(dst: usize, lhs: usize, rhs: usize) -> Ir {
    Ir::BinOp {
        op: BinOp::Add,
        dst,
        lhs,
        rhs,
    }
}

fn program(instructions: Vec<Ir>, register_count: usize) -> IrProgram {
    IrProgram {
        instructions,
        register_count,
    }
}

/// Programs whose answer can be read off.
fn rows() -> Vec<(&'static str, IrProgram, Value)> {
    vec![
        (
            "a register written by what cannot be folded is not the old constant",
            // after the label nothing is known, so `r0 = r2 + r2` is not folded and writes r0
            program(
                vec![
                    int(0, 1),
                    int(2, 5),
                    Ir::Label(0),
                    add(0, 2, 2),
                    Ir::Copy { dst: 1, src: 0 },
                    Ir::Return(1),
                ],
                3,
            ),
            Value::Int(10),
        ),
        (
            "a loop counts: the value at a label is not the one before the loop",
            // i = 0; while i != 3 { i = i + 1 }; return i
            program(
                vec![
                    int(0, 0),
                    int(1, 1),
                    int(2, 3),
                    Ir::Label(0),
                    Ir::BinOp {
                        op: BinOp::Eq,
                        dst: 3,
                        lhs: 0,
                        rhs: 2,
                    },
                    Ir::JumpIfTrue { cond: 3, target: 1 },
                    add(0, 0, 1),
                    Ir::Jump(0),
                    Ir::Label(1),
                    Ir::Return(0),
                ],
                4,
            ),
            Value::Int(3),
        ),
        (
            "two paths meet: the constant of one path is not known after the label",
            // cond is true: the else load is skipped
            program(
                vec![
                    Ir::LoadConst {
                        dst: 0,
                        val: Value::Bool(true),
                    },
                    int(1, 1),
                    Ir::JumpIfTrue { cond: 0, target: 0 },
                    int(1, 2),
                    Ir::Label(0),
                    Ir::Copy { dst: 2, src: 1 },
                    Ir::Return(2),
                ],
                3,
            ),
            Value::Int(1),
        ),
        (
            "a copy from a register with no known value forgets the old constant",
            program(
                vec![
                    Ir::LoadConst {
                        dst: 0,
                        val: Value::Bool(true),
                    },
                    int(1, 1),
                    int(2, 2),
                    Ir::JumpIfTrue { cond: 0, target: 0 },
                    Ir::Label(0),
                    Ir::Copy { dst: 1, src: 2 },
                    Ir::Copy { dst: 3, src: 1 },
                    Ir::Return(3),
                ],
                4,
            ),
            Value::Int(2),
        ),
        (
            "a negation that cannot be folded forgets the old constant",
            program(
                vec![
                    Ir::LoadConst {
                        dst: 0,
                        val: Value::Bool(true),
                    },
                    int(1, 7),
                    int(2, 4),
                    Ir::JumpIfTrue { cond: 0, target: 0 },
                    Ir::Label(0),
                    Ir::UnaryOp {
                        op: UnaryOp::Neg,
                        dst: 1,
                        src: 2,
                    },
                    Ir::Copy { dst: 3, src: 1 },
                    Ir::Return(3),
                ],
                4,
            ),
            Value::Int(-4),
        ),
    ]
}

/// The form the compiler itself produces: both branches of an `if` write one register.
#[test]
fn an_if_keeps_its_value_through_the_optimizer() {
    use alice_compiler::ast::Expr;
    use alice_compiler::compile::{compile_and_run, compile_optimize_and_run};
    let e = Expr::Binary {
        op: BinOp::Add,
        lhs: Box::new(Expr::If {
            cond: Box::new(Expr::Bool(true)),
            then: Box::new(Expr::Int(1)),
            else_: Box::new(Expr::Int(2)),
        }),
        rhs: Box::new(Expr::Int(10)),
    };
    assert_eq!(compile_and_run(&e), Ok(Value::Int(11)));
    assert_eq!(compile_optimize_and_run(&e), Ok(Value::Int(11)));
}

#[test]
fn programs_with_a_known_answer() {
    let rows = rows();
    assert!(rows.len() >= 5);
    let mut wrong = 0;
    for (name, p, expected) in &rows {
        assert_eq!(run(p), Ok(expected.clone()), "as written: {name}");
        for (pass, f) in [
            ("constant_folding", constant_folding as fn(&mut IrProgram)),
            ("constant_propagation", constant_propagation),
            ("dead_code_elimination", dead_code_elimination),
            ("peephole", peephole),
            ("optimize", optimize),
        ] {
            let mut o = p.clone();
            f(&mut o);
            let got = run(&o);
            if got != Ok(expected.clone()) {
                wrong += 1;
                eprintln!("{pass}: {name}: got {got:?}, expected {expected:?}");
            }
        }
    }
    assert_eq!(wrong, 0);
}

/// A fixed sequence (splitmix64), so every run compares the same programs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> usize {
        (self.next() % n) as usize
    }
}

const REGISTERS: usize = 5;
const LABELS: usize = 3;

fn value(r: &mut Rng) -> Value {
    match r.below(8) {
        0 => Value::Bool(true),
        1 => Value::Bool(false),
        2 => Value::Str("ab".to_owned()),
        3 => Value::Float(1.5),
        4 => Value::Int(i64::MIN),
        5 => Value::Int(0),
        _ => Value::Int(r.below(7) as i64 - 3),
    }
}

/// Valid registers and labels. Every label is defined, so the program always passes the check.
fn generated(r: &mut Rng) -> IrProgram {
    const BIN: [BinOp; 9] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Mod,
        BinOp::Eq,
        BinOp::Lt,
        BinOp::And,
        BinOp::Or,
    ];
    let n = 2 + r.below(24);
    let mut ir: Vec<Ir> = (0..n)
        .map(|_| {
            let reg = |r: &mut Rng| r.below(REGISTERS as u64);
            match r.below(12) {
                0..=2 => Ir::LoadConst {
                    dst: reg(r),
                    val: value(r),
                },
                3 | 4 => Ir::Copy {
                    dst: reg(r),
                    src: reg(r),
                },
                5..=7 => Ir::BinOp {
                    op: BIN[r.below(BIN.len() as u64)],
                    dst: reg(r),
                    lhs: reg(r),
                    rhs: reg(r),
                },
                8 => Ir::UnaryOp {
                    op: if r.below(2) == 0 {
                        UnaryOp::Neg
                    } else {
                        UnaryOp::Not
                    },
                    dst: reg(r),
                    src: reg(r),
                },
                9 => Ir::JumpIfFalse {
                    cond: reg(r),
                    target: r.below(LABELS as u64),
                },
                10 => Ir::JumpIfTrue {
                    cond: reg(r),
                    target: r.below(LABELS as u64),
                },
                _ => Ir::Jump(r.below(LABELS as u64)),
            }
        })
        .collect();
    for label in 0..LABELS {
        let at = r.below(ir.len() as u64 + 1);
        ir.insert(at, Ir::Label(label));
    }
    let last = r.below(REGISTERS as u64);
    ir.push(Ir::Return(last));
    program(ir, REGISTERS)
}

#[test]
fn generated_programs_mean_the_same_after_each_pass() {
    let mut rng = Rng(0x4f50_5431);
    let mut compared = 0usize;
    let mut returned = 0usize;
    let mut differ = Vec::new();
    for _ in 0..200_000 {
        let p = generated(&mut rng);
        let plain = run(&p);
        if is_limit(&plain) {
            continue;
        }
        if plain.is_ok() {
            returned += 1;
        }
        for (pass, f) in [
            ("constant_folding", constant_folding as fn(&mut IrProgram)),
            ("constant_propagation", constant_propagation),
            ("dead_code_elimination", dead_code_elimination),
            ("peephole", peephole),
            ("optimize", optimize),
        ] {
            let mut o = p.clone();
            f(&mut o);
            let got = run(&o);
            if is_limit(&got) {
                continue;
            }
            compared += 1;
            if !same(&plain, &got) && differ.len() < 5 {
                differ.push(format!(
                    "{pass}: {plain:?} became {got:?} in {:?}",
                    p.instructions
                ));
            }
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
    assert!(compared > 100_000, "compared {compared}");
    assert!(returned > 10_000, "returned {returned}");
}

#[test]
fn a_program_refused_as_written_is_refused_after_each_pass() {
    // dead code is still part of the program: a bad jump in it is refused before any step runs
    let bad = [
        // a copy of a register to itself is what peephole removes, and this register does not exist
        program(vec![Ir::Copy { dst: 7, src: 7 }], 1),
        program(vec![Ir::Jump(0), Ir::Jump(9), Ir::Label(0)], 0),
        program(
            vec![
                int(0, 1),
                Ir::Jump(0),
                int(7, 1),
                Ir::Label(0),
                Ir::Return(0),
            ],
            1,
        ),
    ];
    for p in &bad {
        assert!(run(p).is_err(), "as written: {p:?}");
        for (pass, f) in [
            ("constant_folding", constant_folding as fn(&mut IrProgram)),
            ("constant_propagation", constant_propagation),
            ("dead_code_elimination", dead_code_elimination),
            ("peephole", peephole),
            ("optimize", optimize),
        ] {
            let mut o = p.clone();
            f(&mut o);
            assert!(run(&o).is_err(), "{pass}: {:?}", o.instructions);
        }
    }
}
