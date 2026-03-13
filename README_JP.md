[English](README.md) | **日本語**

# ALICE-Compiler

[Project A.L.I.C.E.](https://github.com/anthropics/alice) のDSL/JITコンパイラ基盤

## 概要

`alice-compiler` は純Rustによる完全なコンパイラパイプラインです。AST構築からIR変換、最適化パス、レジスタベースVMでの実行までをカバーします。

## 機能

- **AST** — 式ノード（リテラル、変数、二項/単項演算、let束縛、if/else、関数、呼び出し）
- **IR生成** — ASTから中間表現への変換
- **最適化パス** — 定数畳み込み、デッドコード除去、共通部分式除去
- **コード生成** — IRからレジスタベースバイトコードへの出力
- **レジスタVM** — バイトコード実行用仮想マシン
- **型システム** — int、float、bool、string値型
- **二項演算子** — 算術（+, -, *, /, %）、比較（==, !=, <, <=, >, >=）、論理（&&, ||）
- **単項演算子** — 否定（-）、論理否定（!）

## クイックスタート

```rust
use alice_compiler::{Expr, BinOp};

// AST構築: (2 + 3) * 4
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

## アーキテクチャ

```
alice-compiler
├── BinOp / UnaryOp  — 演算子列挙
├── Expr             — AST式ノード
├── IR               — 中間表現
├── Optimizer        — 定数畳み込み、DCE、CSE
├── CodeGen          — IRからバイトコードへの出力
└── VM               — レジスタベース仮想マシン
```

## ライセンス

MIT OR Apache-2.0
