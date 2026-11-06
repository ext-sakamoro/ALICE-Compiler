#!/usr/bin/env python3
"""Breaks the defined results of eval, vm and codegen one way at a time; a test must go red for each.

usage: python3 scripts/compiler_mutants.py
exit 0 if every mutant applied, compiled and made a test report FAILED; exit 1 otherwise.

Run by hand when these files change: each mutant rebuilds the crate. A mutant that does not compile
or no longer applies counts against the result, since it tested nothing. Each file is restored after
its mutant, and at the end even after an interrupt.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMAND = [
    "cargo", "test",
    "--test", "int_arithmetic_wraps",
    "--test", "vm_rejects_a_bad_program",
    "--test", "undefined_variable",
    "--test", "limits",
    "--test", "optimizer_keeps_meaning",
]
FAILED = re.compile(r"^test .* \.\.\. .*FAILED", re.M)

EVAL = "src/eval.rs"
VM = "src/vm.rs"
CODEGEN = "src/codegen.rs"
OPT = "src/optimization.rs"
# (what breaks, file, old text, new text)
MUTANTS = [
    ("negation by the operator", EVAL, "v.wrapping_neg()", "-*v"),
    ("division by the operator", EVAL, "a.wrapping_div(b)", "a / b"),
    ("remainder by the operator", EVAL, "a.wrapping_rem(b)", "a % b"),
    ("programs not checked", VM, "        check_program(program, &label_map)?;\n", ""),
    ("registers allocated infallibly", VM, """        registers
            .try_reserve_exact(program.register_count)
            .map_err(|_| format!("cannot allocate {} registers", program.register_count))?;
""", ""),
    ("the last register is outside", VM, "if r < program.register_count {", "if r + 1 < program.register_count {"),
    ("one past the last register is inside", VM, "if r < program.register_count {", "if r <= program.register_count {"),
    ("a jump target not checked", VM, "Ir::Jump(target) => label(target)?,", "Ir::Jump(_) => {}"),
    ("a conditional target not checked", VM, """                register(*cond)?;
                label(target)?;
""", """                register(*cond)?;
                let _ = target;
"""),
    ("a condition register not checked", VM, """                register(*cond)?;
                label(target)?;
""", """                label(target)?;
                let _ = cond;
"""),
    ("a returned register not checked", VM, "Ir::Return(reg) => register(*reg)?,", "Ir::Return(_) => {}"),
    ("a copied source not checked", VM, """                register(*dst)?;
                register(*src)?;
""", """                register(*dst)?;
                let _ = src;
"""),
    ("a right operand not checked", VM, "                register(*rhs)?;\n", "                let _ = rhs;\n"),
    ("a left operand not checked", VM, "                register(*lhs)?;\n", "                let _ = lhs;\n"),
    ("a loaded register not checked", VM, "Ir::LoadConst { dst, .. } => register(*dst)?,", "Ir::LoadConst { .. } => {}"),
    ("the error names no variable", CODEGEN, "name: name.to_owned(),", "name: String::new(),"),
    ("no step limit", VM, "if steps >= self.limits.max_steps {", "if false {"),
    ("the step limit is one off", VM, "if steps >= self.limits.max_steps {", "if steps > self.limits.max_steps {"),
    ("no string cap", EVAL, "if a.len().saturating_add(b.len()) > MAX_STRING_LEN {", "if false {"),
    ("the string cap is one off", EVAL, "if a.len().saturating_add(b.len()) > MAX_STRING_LEN {", "if a.len().saturating_add(b.len()) >= MAX_STRING_LEN {"),
    ("no total of string bytes", VM, "if *spent > self.limits.max_string_bytes {", "if false {"),
    ("the total is one off", VM, "if *spent > self.limits.max_string_bytes {", "if *spent >= self.limits.max_string_bytes {"),
    ("a load is not charged", VM, "                    self.charge(&mut string_bytes, val)?;\n", ""),
    ("a copy is not charged", VM, "                    self.charge(&mut string_bytes, &self.registers[*src])?;\n", ""),
    ("no register limit", VM, "if program.register_count > self.limits.max_registers {", "if false {"),
    ("the register limit is one off", VM, "if program.register_count > self.limits.max_registers {", "if program.register_count >= self.limits.max_registers {"),
    ("no budget in a pass", OPT, "match s.len().checked_mul(times).filter(|n| *n <= self.0) {", "match Some(0usize) {"),
    ("no depth limit", CODEGEN, "if self.depth >= MAX_EXPR_DEPTH {", "if false {"),
    ("the depth limit is one off", CODEGEN, "if self.depth >= MAX_EXPR_DEPTH {", "if self.depth > MAX_EXPR_DEPTH {"),
    ("folding keeps constants across a label", OPT, """            // another path can come here with other values
            Ir::Label(_) => known.clear(),
            _ => {}
        }
    }
}

/// Constant propagation""", """            _ => {}
        }
    }
}

/// Constant propagation"""),
    ("propagation keeps constants across a label", OPT, """            // another path can come here with other values
            Ir::Label(_) => known.clear(),
            _ => {}
        }
    }
}

/// Dead code""", """            _ => {}
        }
    }
}

/// Dead code"""),
    ("propagation ignores a write by an operation", OPT, """            Ir::BinOp { dst, .. } | Ir::UnaryOp { dst, .. } => {
                known.remove(dst);
            }
""", ""),
    ("dead code removes what would be refused", OPT, """pub fn dead_code_elimination(program: &mut IrProgram) {
    if check_program(program).is_err() {
        return;
    }
""", """pub fn dead_code_elimination(program: &mut IrProgram) {
"""),
    ("peephole removes what would be refused", OPT, """pub fn peephole(program: &mut IrProgram) {
    if check_program(program).is_err() {
        return;
    }
""", """pub fn peephole(program: &mut IrProgram) {
"""),
    ("a let leaves its name bound", CODEGEN, "            self.vars.remove(name);\n", ""),
]


def main() -> int:
    if sys.argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    bad = []
    for name, rel, old, new in MUTANTS:
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            bad.append(name)
            print(f"NOT APPLIED  {name}", flush=True)
            continue
        try:
            path.write_text(original.replace(old, new), encoding="utf-8")
            done = subprocess.run(COMMAND, cwd=ROOT, capture_output=True, text=True, check=False)
        finally:
            path.write_text(original, encoding="utf-8")
        output = done.stdout + done.stderr
        if "could not compile" in output:
            bad.append(name)
            print(f"BROKEN       {name}", flush=True)
        elif done.returncode != 0 and FAILED.search(output):
            print(f"red          {name}", flush=True)
        else:
            bad.append(name)
            print(f"SURVIVED     {name} (rc {done.returncode})", flush=True)
    print(f"{len(MUTANTS) - len(bad)} of {len(MUTANTS)} red")
    return 1 if bad or not MUTANTS else 0


if __name__ == "__main__":
    sys.exit(main())
