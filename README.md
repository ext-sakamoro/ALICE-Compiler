**English** | [日本語](README_JP.md)

# ALICE-Compiler

DSL/JIT compiler infrastructure for [Project A.L.I.C.E.](https://github.com/anthropics/alice)

## Overview

`alice-compiler` provides a complete compiler pipeline in pure Rust — from AST construction through IR lowering, optimization passes, and execution on a register-based VM.

## Features

- **AST** — expression nodes (literals, variables, binary/unary ops, let bindings, if/else, functions, calls)
- **IR Generation** — AST to intermediate representation lowering
- **Optimization Passes** — constant folding, dead code elimination, common subexpression elimination
- **Code Generation** — IR to register-based bytecode emission
- **Register VM** — virtual machine for bytecode execution
- **Type System** — int, float, bool, string value types
- **Binary Operators** — arithmetic (+, -, *, /, %), comparison (==, !=, <, <=, >, >=), logical (&&, ||)
- **Unary Operators** — negation (-), logical not (!)

## Quick Start

```rust
use alice_compiler::{Expr, BinOp};

// Build AST: (2 + 3) * 4
let ast = Expr::Binary {
    op: BinOp::Mul,
    lhs: Box::new(Expr::Binary {
        op: BinOp::Add,
        lhs: Box::new(Expr::Int(2)),
        rhs: Box::new(Expr::Int(3)),
    }),
    rhs: Box::new(Expr::Int(4)),
};
```

## Architecture

```
alice-compiler
├── BinOp / UnaryOp  — operator enums
├── Expr             — AST expression nodes
├── IR               — intermediate representation
├── Optimizer        — constant folding, DCE, CSE
├── CodeGen          — IR to bytecode emission
└── VM               — register-based virtual machine
```

## License

MIT OR Apache-2.0
