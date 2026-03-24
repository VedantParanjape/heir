// 3x3 Matrix Multiplication: C = A * B
// Fully unrolled, row-major flattened inputs: tensor<1x9xi32>
// Layout: A[i][j] = a[0, i*3+j], B[k][j] = b[0, k*3+j]
// C[i][j] = recursive_sum([A[i][k]*B[k][j] for k=0,1,2])
//         = A[i][0]*B[0][j] + (A[i][1]*B[1][j] + A[i][2]*B[2][j])

func.func @matmul_3x3(
    %A: !secret.secret<tensor<1x9xi32>>,
    %B: !secret.secret<tensor<1x9xi32>>
) -> !secret.secret<tensor<1x9xi32>>
    attributes {client.enc_func} {

  %result = secret.generic(
      %A: !secret.secret<tensor<1x9xi32>>,
      %B: !secret.secret<tensor<1x9xi32>>
  ) {
  ^body(%a: tensor<1x9xi32>, %b: tensor<1x9xi32>):

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index

    // A[i][j] = a[0, i*3+j]
    %a00 = tensor.extract %a[%c0, %c0] : tensor<1x9xi32>  // A[0][0]
    %a01 = tensor.extract %a[%c0, %c1] : tensor<1x9xi32>  // A[0][1]
    %a02 = tensor.extract %a[%c0, %c2] : tensor<1x9xi32>  // A[0][2]
    %a10 = tensor.extract %a[%c0, %c3] : tensor<1x9xi32>  // A[1][0]
    %a11 = tensor.extract %a[%c0, %c4] : tensor<1x9xi32>  // A[1][1]
    %a12 = tensor.extract %a[%c0, %c5] : tensor<1x9xi32>  // A[1][2]
    %a20 = tensor.extract %a[%c0, %c6] : tensor<1x9xi32>  // A[2][0]
    %a21 = tensor.extract %a[%c0, %c7] : tensor<1x9xi32>  // A[2][1]
    %a22 = tensor.extract %a[%c0, %c8] : tensor<1x9xi32>  // A[2][2]

    // B[k][j] = b[0, k*3+j]
    %b00 = tensor.extract %b[%c0, %c0] : tensor<1x9xi32>  // B[0][0]
    %b01 = tensor.extract %b[%c0, %c1] : tensor<1x9xi32>  // B[0][1]
    %b02 = tensor.extract %b[%c0, %c2] : tensor<1x9xi32>  // B[0][2]
    %b10 = tensor.extract %b[%c0, %c3] : tensor<1x9xi32>  // B[1][0]
    %b11 = tensor.extract %b[%c0, %c4] : tensor<1x9xi32>  // B[1][1]
    %b12 = tensor.extract %b[%c0, %c5] : tensor<1x9xi32>  // B[1][2]
    %b20 = tensor.extract %b[%c0, %c6] : tensor<1x9xi32>  // B[2][0]
    %b21 = tensor.extract %b[%c0, %c7] : tensor<1x9xi32>  // B[2][1]
    %b22 = tensor.extract %b[%c0, %c8] : tensor<1x9xi32>  // B[2][2]

    // C[0][0] = A[0][0]*B[0][0] + (A[0][1]*B[1][0] + A[0][2]*B[2][0])
    %t00_0 = arith.muli %a00, %b00 : i32
    %t00_1 = arith.muli %a01, %b10 : i32
    %t00_2 = arith.muli %a02, %b20 : i32
    %s00_0 = arith.addi %t00_1, %t00_2 : i32
    %c00   = arith.addi %t00_0, %s00_0 : i32

    // C[0][1] = A[0][0]*B[0][1] + (A[0][1]*B[1][1] + A[0][2]*B[2][1])
    %t01_0 = arith.muli %a00, %b01 : i32
    %t01_1 = arith.muli %a01, %b11 : i32
    %t01_2 = arith.muli %a02, %b21 : i32
    %s01_0 = arith.addi %t01_1, %t01_2 : i32
    %c01   = arith.addi %t01_0, %s01_0 : i32

    // C[0][2] = A[0][0]*B[0][2] + (A[0][1]*B[1][2] + A[0][2]*B[2][2])
    %t02_0 = arith.muli %a00, %b02 : i32
    %t02_1 = arith.muli %a01, %b12 : i32
    %t02_2 = arith.muli %a02, %b22 : i32
    %s02_0 = arith.addi %t02_1, %t02_2 : i32
    %c02   = arith.addi %t02_0, %s02_0 : i32

    // C[1][0] = A[1][0]*B[0][0] + (A[1][1]*B[1][0] + A[1][2]*B[2][0])
    %t10_0 = arith.muli %a10, %b00 : i32
    %t10_1 = arith.muli %a11, %b10 : i32
    %t10_2 = arith.muli %a12, %b20 : i32
    %s10_0 = arith.addi %t10_1, %t10_2 : i32
    %c10   = arith.addi %t10_0, %s10_0 : i32

    // C[1][1] = A[1][0]*B[0][1] + (A[1][1]*B[1][1] + A[1][2]*B[2][1])
    %t11_0 = arith.muli %a10, %b01 : i32
    %t11_1 = arith.muli %a11, %b11 : i32
    %t11_2 = arith.muli %a12, %b21 : i32
    %s11_0 = arith.addi %t11_1, %t11_2 : i32
    %c11   = arith.addi %t11_0, %s11_0 : i32

    // C[1][2] = A[1][0]*B[0][2] + (A[1][1]*B[1][2] + A[1][2]*B[2][2])
    %t12_0 = arith.muli %a10, %b02 : i32
    %t12_1 = arith.muli %a11, %b12 : i32
    %t12_2 = arith.muli %a12, %b22 : i32
    %s12_0 = arith.addi %t12_1, %t12_2 : i32
    %c12   = arith.addi %t12_0, %s12_0 : i32

    // C[2][0] = A[2][0]*B[0][0] + (A[2][1]*B[1][0] + A[2][2]*B[2][0])
    %t20_0 = arith.muli %a20, %b00 : i32
    %t20_1 = arith.muli %a21, %b10 : i32
    %t20_2 = arith.muli %a22, %b20 : i32
    %s20_0 = arith.addi %t20_1, %t20_2 : i32
    %c20   = arith.addi %t20_0, %s20_0 : i32

    // C[2][1] = A[2][0]*B[0][1] + (A[2][1]*B[1][1] + A[2][2]*B[2][1])
    %t21_0 = arith.muli %a20, %b01 : i32
    %t21_1 = arith.muli %a21, %b11 : i32
    %t21_2 = arith.muli %a22, %b21 : i32
    %s21_0 = arith.addi %t21_1, %t21_2 : i32
    %c21   = arith.addi %t21_0, %s21_0 : i32

    // C[2][2] = A[2][0]*B[0][2] + (A[2][1]*B[1][2] + A[2][2]*B[2][2])
    %t22_0 = arith.muli %a20, %b02 : i32
    %t22_1 = arith.muli %a21, %b12 : i32
    %t22_2 = arith.muli %a22, %b22 : i32
    %s22_0 = arith.addi %t22_1, %t22_2 : i32
    %c22   = arith.addi %t22_0, %s22_0 : i32

    // Pack results into tensor<1x9xi32> (row-major)
    %init = arith.constant dense<0> : tensor<1x9xi32>
    %i0 = tensor.insert %c00 into %init[%c0, %c0] : tensor<1x9xi32>
    %i1 = tensor.insert %c01 into %i0[%c0, %c1]   : tensor<1x9xi32>
    %i2 = tensor.insert %c02 into %i1[%c0, %c2]   : tensor<1x9xi32>
    %i3 = tensor.insert %c10 into %i2[%c0, %c3]   : tensor<1x9xi32>
    %i4 = tensor.insert %c11 into %i3[%c0, %c4]   : tensor<1x9xi32>
    %i5 = tensor.insert %c12 into %i4[%c0, %c5]   : tensor<1x9xi32>
    %i6 = tensor.insert %c20 into %i5[%c0, %c6]   : tensor<1x9xi32>
    %i7 = tensor.insert %c21 into %i6[%c0, %c7]   : tensor<1x9xi32>
    %i8 = tensor.insert %c22 into %i7[%c0, %c8]   : tensor<1x9xi32>

    secret.yield %i8 : tensor<1x9xi32>
  } -> !secret.secret<tensor<1x9xi32>>

  return %result : !secret.secret<tensor<1x9xi32>>
}
