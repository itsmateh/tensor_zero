<div align="center">
  <h1>TensorLab </h1>
</div>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Components_stress_tensor.svg" alt="Logo" width="200"/>
</p>

A learning-focused Rust tensor library for exploring linear algebra operations. Built from scratch as a hands-on approach to understanding matrix mathematics and numerical computing.

## 🎯 Project Goal

This library serves as a practical companion to learning linear algebra. Each operation is implemented manually to understand the underlying mechanics of tensor computations.

## ✨ Current Features

### Core Operations
- **Element-wise operations**: Addition, subtraction, Hadamard product (element-wise multiplication)
- **Scalar multiplication**: Multiply tensors by scalar values
- **Matrix multiplication**: Standard matrix product (`matmul`)
- **Operator overloading**: Use `+`, `-`, `*` operators naturally

### Data Management
- Row-major tensor storage with shape validation
- Element access via coordinates
- Comprehensive error handling

## 📖 Quick Start
```rust
use tensor_zero::Tensor;

// Create tensors
let a = Tensor::new(&vec![1.0, 2.0, 3.0, 4.0], &vec![2, 2]).unwrap();
let b = Tensor::new(&vec![5.0, 6.0, 7.0, 8.0], &vec![2, 2]).unwrap();

// Element-wise operations
let sum = (&a + &b).unwrap();     // [6.0, 8.0, 10.0, 12.0]
let diff = (&a - &b).unwrap();    // [-4.0, -4.0, -4.0, -4.0]
let prod = (&a * &b).unwrap();    // [5.0, 12.0, 21.0, 32.0]

// Scalar multiplication
let scaled = a.mul_scalar(2.0);   // [2.0, 4.0, 6.0, 8.0]

// Matrix multiplication
let x = Tensor::new(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &vec![2, 3]).unwrap();
let y = Tensor::new(&vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0], &vec![3, 2]).unwrap();
let result = x.matmul(&y).unwrap(); // Shape: [2, 2]
```

## 🏗️ Project Structure
```
tensor_zero/
├── src/
│   ├── lib.rs                    # Core tensor implementation
│   └── main.rs                   # Example usage
├── tests/
│   └── integration_test.rs       # Test suite
└── Cargo.toml
```

## 🧪 Testing
```bash
cargo test
```

All operations include integration tests covering both valid operations and error cases.

## 🚀 Roadmap

**Next Steps:**
- [X] Matrix transpose
- [X] Optimize matmul with transpose
- [X] Parallel operations with Rayon

**Future Goals:**
- [ ] Broadcasting support
- [ ] N-dimensional operations
- [ ] Reshape and view operations
- [ ] Advanced linear algebra (inverse, determinant, SVD)

## 🔧 Technical Details

- **Storage**: Flat `Vec<f32>` with row-major ordering
- **Operations**: Iterator-based functional style
- **Errors**: Typed errors for shape mismatches and invalid operations
- **Memory**: Zero-copy references where possible

## 🤝 Contributing

This is primarily a learning project, but feel free to suggest improvements or share your own learning journey!


**Note**: Active development follows my linear algebra learning progress. Expect frequent updates as new concepts are explored.