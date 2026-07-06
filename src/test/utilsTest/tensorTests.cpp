#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "Tensor.h"

// Helper function for floating point equality
bool is_close(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// ============================================================================
// 1. CONSTRUCTOR & BASICS TESTS (1D to 4D, zero_grad, offset)
// ============================================================================
void test_constructor_and_basics() {
    std::cout << "Running Test Group 1: Constructor & Basics (1D - 4D)..." << std::endl;

    // Test 1.1: 1D Vector {10}
    Tensor t1({10}, 3.14f);
    assert(t1.size() == 10);
    assert(t1.shape.size() == 1 && t1.shape[0] == 10);
    assert(is_close(t1.data[0], 3.14f) && is_close(t1.data[9], 3.14f));
    assert(t1.offset({5}) == 5);

    // Test 1.2: 2D Matrix {3, 4}
    Tensor t2({3, 4}, -1.0f);
    assert(t2.size() == 12);
    assert(t2.offset({1, 2}) == 1 * 4 + 2); // row 1, col 2 -> index 6
    assert(is_close(t2.data[6], -1.0f));

    // Test 1.3: 3D Tensor {2, 3, 4}
    Tensor t3({2, 3, 4}, 0.5f);
    assert(t3.size() == 24);
    assert(t3.offset({1, 2, 3}) == 1 * (3 * 4) + 2 * 4 + 3); // 12 + 8 + 3 = 23

    // Test 1.4: 4D Tensor {2, 3, 4, 5}
    Tensor t4({2, 3, 4, 5}, 100.0f);
    assert(t4.size() == 120);
    assert(t4.offset({1, 2, 3, 4}) == 1 * (3 * 4 * 5) + 2 * (4 * 5) + 3 * 5 + 4); // 60 + 40 + 15 + 4 = 119
    assert(is_close(t4.data[119], 100.0f));

    // Test 1.5: zero_grad() verification
    t4.grad[0] = 999.0f;
    t4.grad[119] = -50.0f;
    t4.zero_grad();
    for (float g : t4.grad) {
        assert(is_close(g, 0.0f));
    }

    std::cout << "  -> All 5 Constructor & Basics tests passed! ✅\n";
}

// ============================================================================
// 2. ELEMENT-WISE ADDITION TESTS (operator+)
// ============================================================================
void test_addition() {
    std::cout << "Running Test Group 2: Element-wise Addition (operator+)..." << std::endl;

    // Test 2.1: 1D Vectors positive addition
    Tensor a1({5}, 2.0f);
    Tensor b1({5}, 3.0f);
    Tensor c1 = a1 + b1;
    assert(c1.shape == a1.shape && c1.size() == 5);
    for (float val : c1.data) assert(is_close(val, 5.0f));

    // Test 2.2: 2D Matrices negative numbers and zero edge cases
    Tensor a2({3, 3}, -10.0f);
    Tensor b2({3, 3}, 10.0f);
    Tensor c2 = a2 + b2;
    for (float val : c2.data) assert(is_close(val, 0.0f));

    // Test 2.3: 3D Tensors floating point decimals
    Tensor a3({2, 2, 2}, 1.25f);
    Tensor b3({2, 2, 2}, 2.50f);
    Tensor c3 = a3 + b3;
    for (float val : c3.data) assert(is_close(val, 3.75f));

    // Test 2.4: 4D Tensors zero identity (X + 0 = X)
    Tensor a4({2, 2, 2, 2}, 42.0f);
    Tensor b4({2, 2, 2, 2}, 0.0f);
    Tensor c4 = a4 + b4;
    for (size_t i = 0; i < c4.size(); ++i) assert(is_close(c4.data[i], a4.data[i]));

    // Test 2.5: Immutability verification (ensure operands are unchanged)
    a1.data[0] = 100.0f;
    assert(is_close(c1.data[0], 5.0f)); // c1 should not change when a1 changes later

    std::cout << "  -> All 5 Addition tests passed! ✅\n";
}

// ============================================================================
// 3. ELEMENT-WISE MULTIPLICATION TESTS (operator*)
// ============================================================================
void test_multiplication() {
    std::cout << "Running Test Group 3: Element-wise Multiplication (operator*)..." << std::endl;

    // Test 3.1: 1D Vectors positive integers
    Tensor a1({4}, 6.0f);
    Tensor b1({4}, 7.0f);
    Tensor c1 = a1 * b1;
    for (float val : c1.data) assert(is_close(val, 42.0f));

    // Test 3.2: 2D Matrices multiplication by zero identity (X * 0 = 0)
    Tensor a2({3, 4}, 123.456f);
    Tensor b2({3, 4}, 0.0f);
    Tensor c2 = a2 * b2;
    for (float val : c2.data) assert(is_close(val, 0.0f));

    // Test 3.3: 3D Tensors multiplication by one identity (X * 1 = X)
    Tensor a3({2, 3, 4}, -99.9f);
    Tensor b3({2, 3, 4}, 1.0f);
    Tensor c3 = a3 * b3;
    for (size_t i = 0; i < c3.size(); ++i) assert(is_close(c3.data[i], a3.data[i]));

    // Test 3.4: 4D Tensors sign flipping (X * (-1) = -X)
    Tensor a4({2, 2, 2, 2}, 15.0f);
    Tensor b4({2, 2, 2, 2}, -1.0f);
    Tensor c4 = a4 * b4;
    for (float val : c4.data) assert(is_close(val, -15.0f));

    // Test 3.5: Fractional decimals
    Tensor a5({10}, 0.5f);
    Tensor b5({10}, 0.2f);
    Tensor c5 = a5 * b5;
    for (float val : c5.data) assert(is_close(val, 0.1f));

    std::cout << "  -> All 5 Multiplication tests passed! ✅\n";
}

// ============================================================================
// 4. RESHAPE TESTS (1D <-> 2D <-> 3D <-> 4D)
// ============================================================================
void test_reshape() {
    std::cout << "Running Test Group 4: Reshape Transformations..." << std::endl;

    // Create a base tensor with sequential values 0, 1, 2, ..., 23
    Tensor base({24}, 0.0f);
    for (size_t i = 0; i < 24; ++i) {
        base.data[i] = (float)i;
        base.grad[i] = (float)(i * 10);
    }

    // Test 4.1: 1D {24} -> 2D {4, 6}
    Tensor r1 = base.reshape({4, 6});
    assert(r1.shape.size() == 2 && r1.shape[0] == 4 && r1.shape[1] == 6);
    assert(r1.size() == 24);
    assert(is_close(r1.data[r1.offset({2, 3})], base.data[2 * 6 + 3])); // 15.0f

    // Test 4.2: 2D {4, 6} -> 3D {2, 3, 4}
    Tensor r2 = r1.reshape({2, 3, 4});
    assert(r2.shape.size() == 3 && r2.shape[0] == 2 && r2.shape[1] == 3 && r2.shape[2] == 4);
    assert(is_close(r2.data[r2.offset({1, 1, 2})], base.data[1 * 12 + 1 * 4 + 2])); // index 18 -> 18.0f

    // Test 4.3: 3D {2, 3, 4} -> 4D {2, 1, 3, 4}
    Tensor r3 = r2.reshape({2, 1, 3, 4});
    assert(r3.shape.size() == 4 && r3.shape[1] == 1);
    assert(is_close(r3.data[r3.offset({1, 0, 2, 3})], base.data[23])); // index 23 -> 23.0f

    // Test 4.4: 4D -> 1D Flattening
    Tensor r4 = r3.reshape({24});
    assert(r4.shape.size() == 1 && r4.shape[0] == 24);
    assert(is_close(r4.data[10], 10.0f));

    // Test 4.5: Gradient preservation during reshape
    assert(is_close(r4.grad[15], 150.0f));

    std::cout << "  -> All 5 Reshape tests passed! ✅\n";
}

// ============================================================================
// 5. TRANSPOSE TESTS (2D square, rectangular, 3D, 4D attention, involution)
// ============================================================================
void test_transpose() {
    std::cout << "Running Test Group 5: Transpose Transformations..." << std::endl;

    // Test 5.1: 2D Square Matrix {3, 3}
    Tensor m1({3, 3}, 0.0f);
    m1.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor t1 = m1.transpose(0, 1);
    assert(t1.shape[0] == 3 && t1.shape[1] == 3);
    assert(is_close(t1.data[t1.offset({0, 1})], m1.data[m1.offset({1, 0})])); // 4.0f
    assert(is_close(t1.data[t1.offset({2, 0})], m1.data[m1.offset({0, 2})])); // 3.0f

    // Test 5.2: 2D Rectangular Matrix {2, 4} -> {4, 2}
    Tensor m2({2, 4}, 0.0f);
    for (size_t i = 0; i < 8; ++i) m2.data[i] = (float)i;
    Tensor t2 = m2.transpose(0, 1);
    assert(t2.shape[0] == 4 && t2.shape[1] == 2);
    assert(is_close(t2.data[t2.offset({3, 1})], m2.data[m2.offset({1, 3})])); // index 7 -> 7.0f

    // Test 5.3: 3D Tensor {2, 3, 4}, swap Batch (0) and Time (1) -> {3, 2, 4}
    Tensor m3({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < 24; ++i) m3.data[i] = (float)i;
    Tensor t3 = m3.transpose(0, 1);
    assert(t3.shape[0] == 3 && t3.shape[1] == 2 && t3.shape[2] == 4);
    assert(is_close(t3.data[t3.offset({2, 1, 3})], m3.data[m3.offset({1, 2, 3})])); // 23.0f

    // Test 5.4: 4D Attention Tensor {Batch=2, Time=10, Heads=4, Dim=8} -> swap Time(1) and Heads(2)
    Tensor m4({2, 10, 4, 8}, 0.0f);
    for (size_t i = 0; i < m4.size(); ++i) m4.data[i] = (float)i;
    Tensor t4 = m4.transpose(1, 2);
    assert(t4.shape[0] == 2 && t4.shape[1] == 4 && t4.shape[2] == 10 && t4.shape[3] == 8);
    // Coordinate (b=1, h=3, t=9, d=7) in t4 should equal (b=1, t=9, h=3, d=7) in m4!
    assert(is_close(t4.data[t4.offset({1, 3, 9, 7})], m4.data[m4.offset({1, 9, 3, 7})]));

    // Test 5.5: Double Transpose Involution (A^T^T == A)
    Tensor t4_double = t4.transpose(1, 2);
    assert(t4_double.shape == m4.shape);
    for (size_t i = 0; i < m4.size(); ++i) assert(is_close(t4_double.data[i], m4.data[i]));

    std::cout << "  -> All 5 Transpose tests passed! ✅\n";
}

// ============================================================================
// 6. MATRIX MULTIPLICATION TESTS (2D identity, dot product, 3D linear, 3D/4D batched)
// ============================================================================
void test_matmul() {
    std::cout << "Running Test Group 6: Matrix Multiplication (matmul)..." << std::endl;

    // Test 6.1: 2D Identity Multiplication (A * I = A)
    Tensor a1({2, 2}, 0.0f);
    a1.data = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor ident({2, 2}, 0.0f);
    ident.data = {1.0f, 0.0f, 0.0f, 1.0f};
    Tensor c1 = a1.matmul(ident);
    for (size_t i = 0; i < 4; ++i) assert(is_close(c1.data[i], a1.data[i]));

    // Test 6.2: 2D Rectangular Dot Product {2, 3} x {3, 2} -> {2, 2}
    Tensor a2({2, 3}, 0.0f);
    a2.data = {1, 2, 3, 4, 5, 6}; // Row 0: [1, 2, 3], Row 1: [4, 5, 6]
    Tensor b2({3, 2}, 0.0f);
    b2.data = {7, 8, 9, 1, 2, 3}; // Col 0: [7, 9, 2], Col 1: [8, 1, 3]
    Tensor c2 = a2.matmul(b2);
    assert(c2.shape[0] == 2 && c2.shape[1] == 2);
    // c2[0, 0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // c2[0, 1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // c2[1, 0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // c2[1, 1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    assert(is_close(c2.data[0], 31.0f));
    assert(is_close(c2.data[1], 19.0f));
    assert(is_close(c2.data[2], 85.0f));
    assert(is_close(c2.data[3], 55.0f));

    // Test 6.3: 3D Tensor x 2D Weight Matrix (Linear Projection: {Batch=2, Time=3, In=4} x {In=4, Out=5} -> {2, 3, 5})
    Tensor a3({2, 3, 4}, 1.0f);
    Tensor w3({4, 5}, 2.0f);
    Tensor c3 = a3.matmul(w3);
    assert(c3.shape.size() == 3 && c3.shape[0] == 2 && c3.shape[1] == 3 && c3.shape[2] == 5);
    // Each dot product is sum of 4 items of (1.0 * 2.0) = 8.0
    for (float val : c3.data) assert(is_close(val, 8.0f));

    // Test 6.4: Batched 3D x 3D ({Batch=2, M=3, K=4} x {Batch=2, K=4, N=2} -> {2, 3, 2})
    Tensor a4({2, 3, 4}, 1.5f);
    Tensor b4({2, 4, 2}, 2.0f);
    Tensor c4 = a4.matmul(b4);
    assert(c4.shape.size() == 3 && c4.shape[0] == 2 && c4.shape[1] == 3 && c4.shape[2] == 2);
    // Each element is sum of 4 items of (1.5 * 2.0) = 12.0
    for (float val : c4.data) assert(is_close(val, 12.0f));

    // Test 6.5: Batched 4D x 4D (Multi-Head Attention: {B=2, Heads=2, M=3, K=4} x {B=2, Heads=2, K=4, N=3} -> {2, 2, 3, 3})
    Tensor a5({2, 2, 3, 4}, 0.5f);
    Tensor b5({2, 2, 4, 3}, 4.0f);
    Tensor c5 = a5.matmul(b5);
    assert(c5.shape.size() == 4 && c5.shape[0] == 2 && c5.shape[1] == 2 && c5.shape[2] == 3 && c5.shape[3] == 3);
    // Each element is sum of 4 items of (0.5 * 4.0) = 8.0
    for (float val : c5.data) assert(is_close(val, 8.0f));

    std::cout << "  -> All 5 Matmul tests passed! ✅\n";
}

// ============================================================================
// 7. MAP & SCALAR BROADCASTING TESTS (map, scalar arithmetic, bias broadcasting)
// ============================================================================
void test_map_and_scalar() {
    std::cout << "Running Test Group 7: Map & Scalar Broadcasting..." << std::endl;

    // Test 7.1: Unary mapping with custom function (e.g. squaring x -> x^2)
    Tensor t1({5}, 3.0f);
    Tensor sq1 = t1.map([](float x) { return x * x; });
    for (float val : sq1.data) assert(is_close(val, 9.0f));

    // Test 7.2: Scalar addition (t + 5.0f)
    Tensor t2({3, 3}, 10.0f);
    Tensor add2 = t2 + 5.0f;
    for (float val : add2.data) assert(is_close(val, 15.0f));

    // Test 7.3: Scalar multiplication (t * 0.5f)
    Tensor t3({2, 2, 2}, 8.0f);
    Tensor mult3 = t3 * 0.5f;
    for (float val : mult3.data) assert(is_close(val, 4.0f));

    // Test 7.4: Scalar tensor broadcasting in binary arithmetic (t + scalar_tensor)
    Tensor scalar_t({1}, 100.0f);
    Tensor add4 = t3 + scalar_t;
    for (float val : add4.data) assert(is_close(val, 108.0f));

    // Test 7.5: 1D bias broadcasting across 3D batched/time tensor ({2, 3, 4} + {4})
    Tensor batch3d({2, 3, 4}, 1.0f);
    Tensor bias({4}, 0.0f);
    bias.data = {10.0f, 20.0f, 30.0f, 40.0f};
    Tensor out3d = batch3d + bias;
    assert(out3d.shape == batch3d.shape && out3d.size() == 24);
    for (int b = 0; b < 2; ++b) {
        for (int t = 0; t < 3; ++t) {
            assert(is_close(out3d.data[out3d.offset({b, t, 0})], 11.0f));
            assert(is_close(out3d.data[out3d.offset({b, t, 1})], 21.0f));
            assert(is_close(out3d.data[out3d.offset({b, t, 2})], 31.0f));
            assert(is_close(out3d.data[out3d.offset({b, t, 3})], 41.0f));
        }
    }

    std::cout << "  -> All 5 Map & Scalar tests passed! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "       STARTING INDUSTRY COMPLIANT TENSOR ENGINE TESTS       \n";
    std::cout << "============================================================\n\n";

    test_constructor_and_basics();
    test_addition();
    test_multiplication();
    test_reshape();
    test_transpose();
    test_matmul();
    test_map_and_scalar();

    std::cout << "\n============================================================\n";
    std::cout << " 🎉 ALL 35 TENSOR ENGINE TESTS PASSED SUCCESSFULLY! (100%) 🎉\n";
    std::cout << "============================================================\n";
    return 0;
}

