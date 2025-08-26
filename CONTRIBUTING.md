# Contributing to Primus-Turbo

Welcome! We appreciate your interest in contributing to **Primus-Turbo**. This document outlines guidelines and best practices to help you contribute effectively.


## Table of Contents
- [Contributing to Primus-Turbo](#contributing-to-primus-turbo)
  - [Table of Contents](#table-of-contents)
  - [📋 Before You Start](#-before-you-start)
  - [📂 Project Structure](#-project-structure)
  - [🌿 Branch Naming Convention](#-branch-naming-convention)
    - [Type](#type)
    - [Scope (optional)](#scope-optional)
  - [📝 Commit Message Convention](#-commit-message-convention)



## 📋 Before You Start

Before contributing, please:

- Read our README.md to understand the project's goals and architecture.

- Check existing issues and discussions.

- If your contribution is significant (new op, backend integration, design refactor), please open an issue first to discuss your approach.

- Make sure you have a working ROCm environment (ROCm ≥ 6.3 recommended, GFX942/GFX950 tested).


## 📂 Project Structure
```
Primus-Turbo/
├── csrc/                  # Core C++/CUDA/HIP sources
│   ├── include/           # Public headers for kernels & common utilities
│   ├── kernels/           # Core CUDA/HIP kernel implementations
│   ├── pytorch/           # PyTorch C++ bindings for custom operators
│   └── jax/               # JAX C++ bindings (XLA FFI handlers)
├── primus_turbo/          # Python package
│   ├── pytorch/           # PyTorch Python frontend
│   ├── jax/               # JAX Python frontend
│   └── triton/            # Triton kernel implementations
├── tests/                 # Unit & Integration Tests
└── benchmark/             # Performance benchmarks
```


## 🌿 Branch Naming Convention
Please follow this branch naming convention for all feature and bug fix branches:
```
<type>/<scope>/<short-description>
```

### Type
| Type       | Purpose                                     |
| ---------- | ------------------------------------------- |
| `feat`     | New feature or functionality                |
| `opt`      | Performance optimization or tuning          |
| `fix`      | Bug fix                                     |
| `docs`     | Documentation update                        |
| `refactor` | Code refactoring (no functionality change)  |
| `test`     | Tests and test-related changes              |
| `chore`    | Miscellaneous changes (e.g., build scripts) |
| `ci`       | Continuous integration-related changes      |


### Scope (optional)
The scope typically refers to a module, operator, backend, or feature area. Examples:

- `gemm`, `fp8`, `rmsnorm`

- `pytorch`, `jax`, `triton`, `kernels`

- `build`, `docsite`, `bench`

Use your judgment to choose an appropriate scope that improves readability.


## 📝 Commit Message Convention
We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
```
<type>(<scope>): <short description>
```
Good Example:
```
feat(gemm): add fp16/bf16 gemm kernel
opt(fp8): improve quantization performance
fix(attention): correct masking in causal attention
```
Bad Example:
```
update gemm
fix bug
```
