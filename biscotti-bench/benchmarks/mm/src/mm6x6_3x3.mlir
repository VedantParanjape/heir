// Version B: 6x6 as a 2x2 grid of 3x3 blocks.
// Recursion structure identical to scratch.txt (2x2 outer split), but base
// case triggers at n<=3 instead of n<=1, and the base case is a direct
// 3x3 matmul (27 scalar muls, 18 adds).
//
// Call flow:
//   main -> mm(A, B, 6)
//     recursive branch: half=3, hsq=9, 8 recursive calls with n=3
//       each n=3 hits base case -> direct 3x3 matmul
//     pairwise add 4 pairs -> 4 result blocks (3x3 each = 9 elements)
//     assemble into 36-element 6x6 result

!val = i32
!svec = !secret.secret<tensor<?x!val>>

func.func @mm(
    %A  : !svec,
    %B  : !svec,
    %n  : !val { biscotti.progress_argument = 2 }
) -> !svec {

    %c3 = arith.constant 3 : !val
    %c2 = arith.constant 2 : !val

    %cond = arith.cmpi sle, %n, %c3 : !val
    cf.cond_br %cond, ^base, ^recursive { biscotti.base_condition = 0 }

^base:
    // Base case: 3x3 matmul.
    // A, B are tensor<9xi32> (row-major 3x3).
    // C[i,j] = sum_k A[i*3+k] * B[k*3+j] for i,j,k in {0,1,2}.
    %result_base = secret.generic(%A : !svec, %B : !svec) {
        ^bb0(%a: tensor<?x!val>, %b: tensor<?x!val>):
            %c0 = arith.constant 0 : index
            %c1_i = arith.constant 1 : index
            %c2_i = arith.constant 2 : index
            %c3_i = arith.constant 3 : index
            %c4_i = arith.constant 4 : index
            %c5_i = arith.constant 5 : index
            %c6_i = arith.constant 6 : index
            %c7_i = arith.constant 7 : index
            %c8_i = arith.constant 8 : index

            %a0 = tensor.extract %a[%c0]   : tensor<?x!val>
            %a1 = tensor.extract %a[%c1_i] : tensor<?x!val>
            %a2 = tensor.extract %a[%c2_i] : tensor<?x!val>
            %a3 = tensor.extract %a[%c3_i] : tensor<?x!val>
            %a4 = tensor.extract %a[%c4_i] : tensor<?x!val>
            %a5 = tensor.extract %a[%c5_i] : tensor<?x!val>
            %a6 = tensor.extract %a[%c6_i] : tensor<?x!val>
            %a7 = tensor.extract %a[%c7_i] : tensor<?x!val>
            %a8 = tensor.extract %a[%c8_i] : tensor<?x!val>

            %b0 = tensor.extract %b[%c0]   : tensor<?x!val>
            %b1 = tensor.extract %b[%c1_i] : tensor<?x!val>
            %b2 = tensor.extract %b[%c2_i] : tensor<?x!val>
            %b3 = tensor.extract %b[%c3_i] : tensor<?x!val>
            %b4 = tensor.extract %b[%c4_i] : tensor<?x!val>
            %b5 = tensor.extract %b[%c5_i] : tensor<?x!val>
            %b6 = tensor.extract %b[%c6_i] : tensor<?x!val>
            %b7 = tensor.extract %b[%c7_i] : tensor<?x!val>
            %b8 = tensor.extract %b[%c8_i] : tensor<?x!val>

            // C[0,0] = a0*b0 + a1*b3 + a2*b6
            %p00_0 = arith.muli %a0, %b0 : !val
            %p00_1 = arith.muli %a1, %b3 : !val
            %p00_2 = arith.muli %a2, %b6 : !val
            %s00_0 = arith.addi %p00_0, %p00_1 : !val
            %c00 = arith.addi %s00_0, %p00_2 : !val

            // C[0,1] = a0*b1 + a1*b4 + a2*b7
            %p01_0 = arith.muli %a0, %b1 : !val
            %p01_1 = arith.muli %a1, %b4 : !val
            %p01_2 = arith.muli %a2, %b7 : !val
            %s01_0 = arith.addi %p01_0, %p01_1 : !val
            %c01 = arith.addi %s01_0, %p01_2 : !val

            // C[0,2] = a0*b2 + a1*b5 + a2*b8
            %p02_0 = arith.muli %a0, %b2 : !val
            %p02_1 = arith.muli %a1, %b5 : !val
            %p02_2 = arith.muli %a2, %b8 : !val
            %s02_0 = arith.addi %p02_0, %p02_1 : !val
            %c02 = arith.addi %s02_0, %p02_2 : !val

            // C[1,0] = a3*b0 + a4*b3 + a5*b6
            %p10_0 = arith.muli %a3, %b0 : !val
            %p10_1 = arith.muli %a4, %b3 : !val
            %p10_2 = arith.muli %a5, %b6 : !val
            %s10_0 = arith.addi %p10_0, %p10_1 : !val
            %c10 = arith.addi %s10_0, %p10_2 : !val

            // C[1,1] = a3*b1 + a4*b4 + a5*b7
            %p11_0 = arith.muli %a3, %b1 : !val
            %p11_1 = arith.muli %a4, %b4 : !val
            %p11_2 = arith.muli %a5, %b7 : !val
            %s11_0 = arith.addi %p11_0, %p11_1 : !val
            %c11 = arith.addi %s11_0, %p11_2 : !val

            // C[1,2] = a3*b2 + a4*b5 + a5*b8
            %p12_0 = arith.muli %a3, %b2 : !val
            %p12_1 = arith.muli %a4, %b5 : !val
            %p12_2 = arith.muli %a5, %b8 : !val
            %s12_0 = arith.addi %p12_0, %p12_1 : !val
            %c12 = arith.addi %s12_0, %p12_2 : !val

            // C[2,0] = a6*b0 + a7*b3 + a8*b6
            %p20_0 = arith.muli %a6, %b0 : !val
            %p20_1 = arith.muli %a7, %b3 : !val
            %p20_2 = arith.muli %a8, %b6 : !val
            %s20_0 = arith.addi %p20_0, %p20_1 : !val
            %c20 = arith.addi %s20_0, %p20_2 : !val

            // C[2,1] = a6*b1 + a7*b4 + a8*b7
            %p21_0 = arith.muli %a6, %b1 : !val
            %p21_1 = arith.muli %a7, %b4 : !val
            %p21_2 = arith.muli %a8, %b7 : !val
            %s21_0 = arith.addi %p21_0, %p21_1 : !val
            %c21 = arith.addi %s21_0, %p21_2 : !val

            // C[2,2] = a6*b2 + a7*b5 + a8*b8
            %p22_0 = arith.muli %a6, %b2 : !val
            %p22_1 = arith.muli %a7, %b5 : !val
            %p22_2 = arith.muli %a8, %b8 : !val
            %s22_0 = arith.addi %p22_0, %p22_1 : !val
            %c22 = arith.addi %s22_0, %p22_2 : !val

            %out = tensor.from_elements
                %c00, %c01, %c02,
                %c10, %c11, %c12,
                %c20, %c21, %c22 : tensor<9x!val>
            %out_cast = tensor.cast %out : tensor<9x!val> to tensor<?x!val>
            secret.yield %out_cast : tensor<?x!val>
    } -> (!svec)
    return %result_base : !svec

^recursive:
    %half = arith.divsi %n, %c2 : !val
    %hsq = arith.muli %half, %half : !val
    %half_idx = arith.index_cast %half : !val to index
    %hsq_idx = arith.index_cast %hsq : !val to index
    %n_idx = arith.index_cast %n : !val to index

    // Quadrant extraction: identical to scratch.txt.
    %a00 = secret.generic(%A : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a01 = secret.generic(%A : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %src2 = arith.addi %src1, %h : index
                    %v = tensor.extract %a[%src2] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a10 = secret.generic(%A : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %row1 = arith.addi %row, %h : index
                    %src = arith.muli %row1, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a11 = secret.generic(%A : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %row1 = arith.addi %row, %h : index
                    %src = arith.muli %row1, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %src2 = arith.addi %src1, %h : index
                    %v = tensor.extract %a[%src2] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b00 = secret.generic(%B : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%b: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %b[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b01 = secret.generic(%B : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%b: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %src2 = arith.addi %src1, %h : index
                    %v = tensor.extract %b[%src2] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b10 = secret.generic(%B : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%b: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %row1 = arith.addi %row, %h : index
                    %src = arith.muli %row1, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %b[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b11 = secret.generic(%B : !svec, %half_idx : index, %n_idx : index, %hsq_idx : index) {
        ^bb0(%b: tensor<?x!val>, %h: index, %nn: index, %hh: index):
            %out = tensor.generate %hh {
                ^bb0(%i: index):
                    %row = arith.divui %i, %h : index
                    %col = arith.remui %i, %h : index
                    %row1 = arith.addi %row, %h : index
                    %src = arith.muli %row1, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %src2 = arith.addi %src1, %h : index
                    %v = tensor.extract %b[%src2] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    // 8 recursive calls on (half x half)-sized tensors. Each hits base case.
    %t0 = call @mm(%a00, %b00, %half) { biscotti.recursive_call = 0 } : (!svec, !svec, !val) -> !svec
    %t1 = call @mm(%a01, %b10, %half) { biscotti.recursive_call = 1 } : (!svec, !svec, !val) -> !svec
    %t2 = call @mm(%a00, %b01, %half) { biscotti.recursive_call = 2 } : (!svec, !svec, !val) -> !svec
    %t3 = call @mm(%a01, %b11, %half) { biscotti.recursive_call = 3 } : (!svec, !svec, !val) -> !svec
    %t4 = call @mm(%a10, %b00, %half) { biscotti.recursive_call = 4 } : (!svec, !svec, !val) -> !svec
    %t5 = call @mm(%a11, %b10, %half) { biscotti.recursive_call = 5 } : (!svec, !svec, !val) -> !svec
    %t6 = call @mm(%a10, %b01, %half) { biscotti.recursive_call = 6 } : (!svec, !svec, !val) -> !svec
    %t7 = call @mm(%a11, %b11, %half) { biscotti.recursive_call = 7 } : (!svec, !svec, !val) -> !svec

    // Pairwise add (each is half*half sized).
    %c00 = secret.generic(%t0 : !svec, %t1 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>):
            %s = arith.addi %x, %y : tensor<?x!val>
            secret.yield %s : tensor<?x!val>
    } -> (!svec)
    %c01 = secret.generic(%t2 : !svec, %t3 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>):
            %s = arith.addi %x, %y : tensor<?x!val>
            secret.yield %s : tensor<?x!val>
    } -> (!svec)
    %c10 = secret.generic(%t4 : !svec, %t5 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>):
            %s = arith.addi %x, %y : tensor<?x!val>
            secret.yield %s : tensor<?x!val>
    } -> (!svec)
    %c11 = secret.generic(%t6 : !svec, %t7 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>):
            %s = arith.addi %x, %y : tensor<?x!val>
            secret.yield %s : tensor<?x!val>
    } -> (!svec)

    // Assemble four (half x half) quadrants into (n x n) result.
    %nsq = arith.muli %n, %n : !val
    %nsq_idx = arith.index_cast %nsq : !val to index

    %result = secret.generic(
        %c00 : !svec, %c01 : !svec, %c10 : !svec, %c11 : !svec,
        %half_idx : index, %n_idx : index, %nsq_idx : index
    ) {
        ^bb0(%q00: tensor<?x!val>, %q01: tensor<?x!val>,
             %q10: tensor<?x!val>, %q11: tensor<?x!val>,
             %h: index, %nn: index, %nn2: index):
            %out = tensor.generate %nn2 {
                ^bb0(%i: index):
                    %row = arith.divui %i, %nn : index
                    %col = arith.remui %i, %nn : index
                    %row_local = arith.remui %row, %h : index
                    %col_local = arith.remui %col, %h : index
                    %local_idx = arith.muli %row_local, %h : index
                    %local_idx1 = arith.addi %local_idx, %col_local : index

                    %v00 = tensor.extract %q00[%local_idx1] : tensor<?x!val>
                    %v01 = tensor.extract %q01[%local_idx1] : tensor<?x!val>
                    %v10 = tensor.extract %q10[%local_idx1] : tensor<?x!val>
                    %v11 = tensor.extract %q11[%local_idx1] : tensor<?x!val>

                    // For n=6, h=3 → row/col in {0..5}, is_top = row<3, is_left = col<3.
                    // (Fixed vs scratch.txt: use ult against %h, not eq against zero,
                    // so this generalises past the power-of-2 case.)
                    %is_top  = arith.cmpi ult, %row, %h : index
                    %is_left = arith.cmpi ult, %col, %h : index

                    %top_val = arith.select %is_left, %v00, %v01 : !val
                    %bot_val = arith.select %is_left, %v10, %v11 : !val
                    %val = arith.select %is_top, %top_val, %bot_val : !val

                    tensor.yield %val : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    return %result : !svec
}

!inputvec = !secret.secret<tensor<36xi32>>
func.func @main(%A : !inputvec, %B : !inputvec) -> !inputvec {
    %c6 = arith.constant 6 : !val

    %A_dyn = secret.generic(%A : !inputvec) {
    ^bb0(%a: tensor<36xi32>):
        %cast = tensor.cast %a : tensor<36xi32> to tensor<?xi32>
        secret.yield %cast : tensor<?xi32>
    } -> !svec

    %B_dyn = secret.generic(%B : !inputvec) {
    ^bb0(%b: tensor<36xi32>):
        %cast = tensor.cast %b : tensor<36xi32> to tensor<?xi32>
        secret.yield %cast : tensor<?xi32>
    } -> !svec

    %result = call @mm(%A_dyn, %B_dyn, %c6)
        { biscotti.call = 0 } : (!svec, !svec, !val) -> !svec

    %result_static = secret.generic(%result : !svec) {
    ^bb0(%r: tensor<?xi32>):
        %cast = tensor.cast %r : tensor<?xi32> to tensor<36xi32>
        secret.yield %cast : tensor<36xi32>
    } -> !inputvec

    return %result_static : !inputvec
}
