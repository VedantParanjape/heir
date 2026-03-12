// 2x2 Matrix Multiplication: C = A * B
// Fully unrolled, row-major flattened inputs: tensor<1x4xi32>
// Layout: [row0col0, row0col1, row1col0, row1col1]
// C[i][j] = sum_k A[i][k] * B[k][j]

func.func @matmul_2x2(
    %A: !secret.secret<tensor<1x4xi32>>,
    %B: !secret.secret<tensor<1x4xi32>>
) -> !secret.secret<tensor<1x4xi32>>
    attributes {client.enc_func} {

  %result = secret.generic(
      %A: !secret.secret<tensor<1x4xi32>>,
      %B: !secret.secret<tensor<1x4xi32>>
  ) {
  ^body(%a: tensor<1x4xi32>, %b: tensor<1x4xi32>):

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Row-major: A[i][k] = a[0, i*2+k], B[k][j] = b[0, k*2+j]
    %a00 = tensor.extract %a[%c0, %c0] : tensor<1x4xi32>  // A[0][0]
    %a01 = tensor.extract %a[%c0, %c1] : tensor<1x4xi32>  // A[0][1]
    %a10 = tensor.extract %a[%c0, %c2] : tensor<1x4xi32>  // A[1][0]
    %a11 = tensor.extract %a[%c0, %c3] : tensor<1x4xi32>  // A[1][1]

    %b00 = tensor.extract %b[%c0, %c0] : tensor<1x4xi32>  // B[0][0]
    %b01 = tensor.extract %b[%c0, %c1] : tensor<1x4xi32>  // B[0][1]
    %b10 = tensor.extract %b[%c0, %c2] : tensor<1x4xi32>  // B[1][0]
    %b11 = tensor.extract %b[%c0, %c3] : tensor<1x4xi32>  // B[1][1]

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

    // Pack results into tensor<1x4xi32> (row-major)
    %init = arith.constant dense<0> : tensor<1x4xi32>
    %i0 = tensor.insert %c00 into %init[%c0, %c0] : tensor<1x4xi32>
    %i1 = tensor.insert %c01 into %i0[%c0, %c1]   : tensor<1x4xi32>
    %i2 = tensor.insert %c10 into %i1[%c0, %c2]   : tensor<1x4xi32>
    %i3 = tensor.insert %c11 into %i2[%c0, %c3]   : tensor<1x4xi32>

    secret.yield %i3 : tensor<1x4xi32>
  } -> !secret.secret<tensor<1x4xi32>>

  return %result : !secret.secret<tensor<1x4xi32>>
}
