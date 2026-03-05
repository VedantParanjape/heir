// 2x2 Matrix Multiplication: C = A * B
// Fully unrolled — each operation is written manually.
// C[i][j] = sum_k A[i][k] * B[k][j]

func.func @matmul_2x2(
    %A: !secret.secret<tensor<2x2xi32>>,
    %B: !secret.secret<tensor<2x2xi32>>
) -> !secret.secret<tensor<2x2xi32>>
    attributes {client.enc_func} {

  %result = secret.generic(
      %A: !secret.secret<tensor<2x2xi32>>,
      %B: !secret.secret<tensor<2x2xi32>>
  ) {
  ^body(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>):

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %a00 = tensor.extract %a[%c0, %c0] : tensor<2x2xi32>
    %a01 = tensor.extract %a[%c0, %c1] : tensor<2x2xi32>
    %a10 = tensor.extract %a[%c1, %c0] : tensor<2x2xi32>
    %a11 = tensor.extract %a[%c1, %c1] : tensor<2x2xi32>

    %b00 = tensor.extract %b[%c0, %c0] : tensor<2x2xi32>
    %b01 = tensor.extract %b[%c0, %c1] : tensor<2x2xi32>
    %b10 = tensor.extract %b[%c1, %c0] : tensor<2x2xi32>
    %b11 = tensor.extract %b[%c1, %c1] : tensor<2x2xi32>

    // C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    %t0  = arith.muli %a00, %b00 : i32
    %t1  = arith.muli %a01, %b10 : i32
    %c00 = arith.addi %t0,  %t1  : i32

    // C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    %t2  = arith.muli %a00, %b01 : i32
    %t3  = arith.muli %a01, %b11 : i32
    %c01 = arith.addi %t2,  %t3  : i32

    // C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    %t4  = arith.muli %a10, %b00 : i32
    %t5  = arith.muli %a11, %b10 : i32
    %c10 = arith.addi %t4,  %t5  : i32

    // C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
    %t6  = arith.muli %a10, %b01 : i32
    %t7  = arith.muli %a11, %b11 : i32
    %c11 = arith.addi %t6,  %t7  : i32

    %init = arith.constant dense<0> : tensor<2x2xi32>
    %i0 = tensor.insert %c00 into %init[%c0, %c0] : tensor<2x2xi32>
    %i1 = tensor.insert %c01 into %i0[%c0, %c1]   : tensor<2x2xi32>
    %i2 = tensor.insert %c10 into %i1[%c1, %c0]   : tensor<2x2xi32>
    %i3 = tensor.insert %c11 into %i2[%c1, %c1]   : tensor<2x2xi32>

    secret.yield %i3 : tensor<2x2xi32>
  } -> !secret.secret<tensor<2x2xi32>>

  return %result : !secret.secret<tensor<2x2xi32>>
}
