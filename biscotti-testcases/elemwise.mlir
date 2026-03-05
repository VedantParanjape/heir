// Element-wise multiplication of two 8-element vectors: C[i] = A[i] * B[i]
// Fully unrolled — each multiply is written manually.

func.func @elementwise_mul(
    %A: !secret.secret<tensor<8xi32>>,
    %B: !secret.secret<tensor<8xi32>>
) -> !secret.secret<tensor<8xi32>>
    attributes {client.enc_func} {

  %result = secret.generic(
      %A: !secret.secret<tensor<8xi32>>,
      %B: !secret.secret<tensor<8xi32>>
  ) {
  ^body(%a: tensor<8xi32>, %b: tensor<8xi32>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index

    %a0 = tensor.extract %a[%c0] : tensor<8xi32>
    %a1 = tensor.extract %a[%c1] : tensor<8xi32>
    %a2 = tensor.extract %a[%c2] : tensor<8xi32>
    %a3 = tensor.extract %a[%c3] : tensor<8xi32>
    %a4 = tensor.extract %a[%c4] : tensor<8xi32>
    %a5 = tensor.extract %a[%c5] : tensor<8xi32>
    %a6 = tensor.extract %a[%c6] : tensor<8xi32>
    %a7 = tensor.extract %a[%c7] : tensor<8xi32>

    %b0 = tensor.extract %b[%c0] : tensor<8xi32>
    %b1 = tensor.extract %b[%c1] : tensor<8xi32>
    %b2 = tensor.extract %b[%c2] : tensor<8xi32>
    %b3 = tensor.extract %b[%c3] : tensor<8xi32>
    %b4 = tensor.extract %b[%c4] : tensor<8xi32>
    %b5 = tensor.extract %b[%c5] : tensor<8xi32>
    %b6 = tensor.extract %b[%c6] : tensor<8xi32>
    %b7 = tensor.extract %b[%c7] : tensor<8xi32>

    %r0 = arith.muli %a0, %b0 : i32
    %r1 = arith.muli %a1, %b1 : i32
    %r2 = arith.muli %a2, %b2 : i32
    %r3 = arith.muli %a3, %b3 : i32
    %r4 = arith.muli %a4, %b4 : i32
    %r5 = arith.muli %a5, %b5 : i32
    %r6 = arith.muli %a6, %b6 : i32
    %r7 = arith.muli %a7, %b7 : i32

    %init = arith.constant dense<0> : tensor<8xi32>
    %i0 = tensor.insert %r0 into %init[%c0] : tensor<8xi32>
    %i1 = tensor.insert %r1 into %i0[%c1]   : tensor<8xi32>
    %i2 = tensor.insert %r2 into %i1[%c2]   : tensor<8xi32>
    %i3 = tensor.insert %r3 into %i2[%c3]   : tensor<8xi32>
    %i4 = tensor.insert %r4 into %i3[%c4]   : tensor<8xi32>
    %i5 = tensor.insert %r5 into %i4[%c5]   : tensor<8xi32>
    %i6 = tensor.insert %r6 into %i5[%c6]   : tensor<8xi32>
    %i7 = tensor.insert %r7 into %i6[%c7]   : tensor<8xi32>

    secret.yield %i7 : tensor<8xi32>
  } -> !secret.secret<tensor<8xi32>>

  return %result : !secret.secret<tensor<8xi32>>
}
