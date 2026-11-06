//! An untrusted program ends, and holds a stated amount of memory, on any input.
//!
//! oracle: steps, string bytes and registers are counted, so each limit is exact and the same on every
//! machine: a program at a limit runs and one past it is an `Err`. The peak is measured by a counting
//! allocator and compared with what the limits allow, not with what the code happens to use.
//! The forms are those of the fuzz findings: `Label 0; Jump 0` (timeout in `fuzz_ir_execute`) and
//! `s = s + s` repeated (out of memory in `fuzz_optimize_passes`, a 139-byte input), plus the form the
//! per-string cap alone would not stop: many strings at that cap, kept in registers.

use alice_compiler::ast::{BinOp, Expr};
use alice_compiler::codegen::{Codegen, CompileError, MAX_EXPR_DEPTH};
use alice_compiler::ir::{Ir, IrProgram, Value};
use alice_compiler::optimization::{
    constant_folding, constant_propagation, optimize, MAX_PASS_STRING_BYTES,
};
use alice_compiler::vm::{Vm, VmLimits};
use alice_compiler::MAX_STRING_LEN;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const REFUSE_ABOVE: usize = 1 << 30;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // far above any limit: without a limit, this is where the test ends, not the machine's memory
        if LIVE.load(Ordering::Relaxed) + l.size() > REFUSE_ABOVE {
            return std::ptr::null_mut();
        }
        let n = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        PEAK.fetch_max(n, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The peak is one counter: these tests run one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Runs `f` and returns what it allocated above the level it started at, at the most.
fn peak_of(f: impl FnOnce()) -> usize {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    f();
    PEAK.load(Ordering::Relaxed).saturating_sub(before)
}

const MIB: usize = 1 << 20;
/// What the default limits allow in one run: the string bytes, the registers themselves, and 2 MiB
/// for the rest. Measured at 89.0 MiB with every register allocated and every string byte made.
const STATED_PEAK: usize = VmLimits::DEFAULT.max_string_bytes
    + VmLimits::DEFAULT.max_registers * std::mem::size_of::<Value>()
    + 2 * MIB;
/// What one pass allows: its string bytes, and 2 MiB for the rest. A pass allocates no registers.
/// `optimize` runs two passes that make strings, measured at 94.5 MiB together.
const STATED_PASS_PEAK: usize = MAX_PASS_STRING_BYTES + 2 * MIB;

fn program(instructions: Vec<Ir>, register_count: usize) -> IrProgram {
    IrProgram {
        instructions,
        register_count,
    }
}

fn load(dst: usize, s: &str) -> Ir {
    Ir::LoadConst {
        dst,
        val: Value::Str(s.to_owned()),
    }
}

fn double(reg: usize) -> Ir {
    Ir::BinOp {
        op: BinOp::Add,
        dst: reg,
        lhs: reg,
        rhs: reg,
    }
}

/// `s = s + s` as many times as it takes to go past any cap.
fn doubling() -> IrProgram {
    let mut ir = vec![load(0, "vvvacapv")];
    ir.extend((0..64).map(|_| double(0)));
    ir.push(Ir::Return(0));
    program(ir, 1)
}

#[test]
fn a_loop_ends_at_the_step_limit() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let endless = program(vec![Ir::Label(0), Ir::Jump(0)], 0);
    let err = Vm::new().execute(&endless);
    assert!(err.as_ref().is_err_and(|e| e.contains("steps")), "{err:?}");
}

#[test]
fn the_step_limit_is_exact() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let three = program(vec![Ir::Nop, Ir::Nop, Ir::Nop], 0);
    let limits = |max_steps| VmLimits {
        max_steps,
        ..VmLimits::DEFAULT
    };
    assert_eq!(Vm::with_limits(limits(3)).execute(&three), Ok(Value::Unit));
    assert!(Vm::with_limits(limits(2)).execute(&three).is_err());
}

#[test]
fn a_doubling_string_stops_at_its_cap_in_the_vm() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let mut result = Ok(Value::Unit);
    let peak = peak_of(|| result = Vm::new().execute(&doubling()));
    assert!(result.is_err(), "{result:?}");
    assert!(peak < STATED_PEAK, "{peak}");
}

#[test]
fn a_doubling_string_stops_at_its_cap_in_the_folder() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let mut p = doubling();
    let peak = peak_of(|| optimize(&mut p));
    assert!(peak < STATED_PEAK, "{peak}");
    // what it could not fold stays in the program, and meets the same cap at run time
    assert!(Vm::new().execute(&p).is_err());
}

#[test]
fn the_longest_string_fits_and_one_byte_more_does_not() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let half = "x".repeat(MAX_STRING_LEN / 2);
    let add = |a: &str, b: &str| {
        program(
            vec![
                load(0, a),
                load(1, b),
                Ir::BinOp {
                    op: BinOp::Add,
                    dst: 2,
                    lhs: 0,
                    rhs: 1,
                },
                Ir::Return(2),
            ],
            3,
        )
    };
    let fits = Vm::new().execute(&add(&half, &half));
    assert!(matches!(fits, Ok(Value::Str(ref s)) if s.len() == MAX_STRING_LEN));
    assert!(Vm::new().execute(&add(&half, &format!("{half}x"))).is_err());
}

/// A string at the cap, then copies of it into `copies` registers.
fn many_capped_strings(copies: usize) -> IrProgram {
    let mut ir = vec![load(0, &"x".repeat(MAX_STRING_LEN))];
    ir.extend((1..=copies).map(|dst| Ir::Copy { dst, src: 0 }));
    ir.push(Ir::Return(0));
    program(ir, copies + 1)
}

#[test]
fn many_strings_at_the_cap_stop_at_the_total() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    // 4096 copies of 1 MiB would be 4 GiB
    let p = many_capped_strings(4096);
    let mut result = Ok(Value::Unit);
    let peak = peak_of(|| result = Vm::new().execute(&p));
    assert!(
        result.as_ref().is_err_and(|e| e.contains("string bytes")),
        "{result:?}"
    );
    assert!(peak < STATED_PEAK, "{peak}");
}

#[test]
fn the_stated_peak_holds_with_every_register_allocated() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let mut p = many_capped_strings(4096);
    p.register_count = VmLimits::DEFAULT.max_registers;
    let mut result = Ok(Value::Unit);
    let peak = peak_of(|| result = Vm::new().execute(&p));
    assert!(result.is_err(), "{result:?}");
    assert!(peak < STATED_PEAK, "{peak}");
    // the registers and the strings are both there: the statement is about their sum
    assert!(peak > VmLimits::DEFAULT.max_string_bytes, "{peak}");
}

#[test]
fn the_same_strings_made_again_and_again_stop_at_the_total() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    // nothing is kept, so only a count of what was made can end this before the steps run out
    let p = program(
        vec![
            load(0, &"x".repeat(MAX_STRING_LEN)),
            Ir::Label(0),
            Ir::Copy { dst: 1, src: 0 },
            Ir::Jump(0),
        ],
        2,
    );
    let err = Vm::new().execute(&p);
    assert!(
        err.as_ref().is_err_and(|e| e.contains("string bytes")),
        "{err:?}"
    );
}

#[test]
fn the_string_bytes_limit_is_exact() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let limits = |max_string_bytes| VmLimits {
        max_string_bytes,
        ..VmLimits::DEFAULT
    };
    // a 3-byte load and a copy of it
    let p = program(
        vec![load(0, "abc"), Ir::Copy { dst: 1, src: 0 }, Ir::Return(1)],
        2,
    );
    assert!(Vm::with_limits(limits(6)).execute(&p).is_ok());
    assert!(Vm::with_limits(limits(5)).execute(&p).is_err());
}

#[test]
fn each_folding_pass_stays_under_the_total() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    for (name, pass, passes) in [
        ("folding", constant_folding as fn(&mut IrProgram), 1),
        ("propagation", constant_propagation, 1),
        ("all", optimize, 2),
    ] {
        let mut p = many_capped_strings(4096);
        let peak = peak_of(|| pass(&mut p));
        assert!(peak < passes * STATED_PASS_PEAK, "{name}: {peak}");
    }
}

#[test]
fn the_result_of_an_addition_counts_towards_the_total() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let limits = |max_string_bytes| VmLimits {
        max_string_bytes,
        ..VmLimits::DEFAULT
    };
    // two 3-byte loads and their 6-byte sum
    let p = program(
        vec![
            load(0, "abc"),
            load(1, "def"),
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
    assert!(Vm::with_limits(limits(12)).execute(&p).is_ok());
    assert!(Vm::with_limits(limits(11)).execute(&p).is_err());
}

/// More strings than a pass may make: 128 copies of 1 MiB, in the form that goes through one charge.
fn distinct_loads() -> IrProgram {
    let ir = (0..128).map(|dst| load(dst, &"x".repeat(MIB))).collect();
    program(ir, 128)
}

fn distinct_sums() -> IrProgram {
    let half = "x".repeat(MIB / 2);
    let mut ir = vec![load(0, &half), load(1, &half)];
    ir.extend((2..130).map(|dst| Ir::BinOp {
        op: BinOp::Add,
        dst,
        lhs: 0,
        rhs: 1,
    }));
    program(ir, 130)
}

#[test]
fn every_place_a_pass_makes_a_string_is_counted() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    for (name, pass, make) in [
        (
            "folding, loads",
            constant_folding as fn(&mut IrProgram),
            distinct_loads as fn() -> IrProgram,
        ),
        ("folding, sums", constant_folding, distinct_sums),
        ("folding, copies", constant_folding, || {
            many_capped_strings(128)
        }),
        ("propagation, loads", constant_propagation, distinct_loads),
        ("propagation, copies", constant_propagation, || {
            many_capped_strings(128)
        }),
    ] {
        let mut p = make();
        let peak = peak_of(|| pass(&mut p));
        assert!(peak < STATED_PASS_PEAK, "{name}: {peak}");
        // the program is large enough that the count is what stopped it
        assert!(peak > STATED_PASS_PEAK / 2, "{name}: {peak}");
    }
}

#[test]
fn folding_short_strings_still_folds() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let mut p = program(
        vec![
            load(0, "ab"),
            load(1, "cd"),
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
    optimize(&mut p);
    assert_eq!(p.instructions[2], load(2, "abcd"), "{:?}", p.instructions);
}

#[test]
fn registers_over_the_limit_are_refused_before_they_are_allocated() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let limit = VmLimits::DEFAULT.max_registers;
    let mut result = Ok(Value::Unit);
    let peak = peak_of(|| result = Vm::new().execute(&program(vec![], limit + 1)));
    assert!(result.is_err());
    assert!(peak < MIB, "{peak}");
    assert_eq!(Vm::new().execute(&program(vec![], limit)), Ok(Value::Unit));
    // a caller's own limit can allow what no allocator can give: an error too, not a panic
    let no_limit = VmLimits {
        max_registers: usize::MAX,
        ..VmLimits::DEFAULT
    };
    assert!(Vm::with_limits(no_limit)
        .execute(&program(vec![], usize::MAX))
        .is_err());
}

/// `-(-(-(... 1)))` nested `depth` levels, built without recursion.
fn nested(depth: usize) -> Expr {
    let mut e = Expr::Int(1);
    for _ in 1..depth {
        e = Expr::Unary {
            op: alice_compiler::ast::UnaryOp::Neg,
            operand: Box::new(e),
        };
    }
    e
}

#[test]
fn nesting_stops_at_the_limit() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    assert!(Codegen::new().compile(&nested(MAX_EXPR_DEPTH)).is_ok());
    assert_eq!(
        Codegen::new()
            .compile(&nested(MAX_EXPR_DEPTH + 1))
            .map(|_| ()),
        Err(CompileError::TooDeep)
    );
    // deep enough to overflow the stack of this thread if lowering recursed through it; dropping an
    // `Expr` recurses too, which is the caller's own structure, so this one is not dropped
    let deep = nested(1_000_000);
    assert_eq!(
        Codegen::new().compile(&deep).map(|_| ()),
        Err(CompileError::TooDeep)
    );
    std::mem::forget(deep);
}

#[test]
fn nesting_counts_depth_and_not_breadth() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let wide = Expr::Block(vec![Expr::Int(1); MAX_EXPR_DEPTH * 2]);
    assert!(Codegen::new().compile_expr(&wide).is_ok());
}

#[test]
fn the_limit_of_nesting_fits_a_spawned_thread() {
    let _one = ONE_AT_A_TIME.lock().unwrap();
    let e = nested(MAX_EXPR_DEPTH);
    std::thread::Builder::new()
        .stack_size(2 * MIB)
        .spawn(move || {
            assert!(Codegen::new().compile(&e).is_ok());
            drop(e);
        })
        .unwrap()
        .join()
        .unwrap();
}
