// Version A: 6x6 as a 3x3 grid of 2x2 blocks.
// Recursion structure: outer 3x3 block decomposition at n=6, base case at
// n<=2 (direct 2x2 kernel). This function is meaningful for n=6 only —
// the recursive branch hardcodes a 3-way split.
//
// Call flow:
//   main -> mm(A, B, 6)
//     recursive branch: b=2, 27 recursive calls with n=2
//       each n=2 hits base case -> direct 2x2 matmul (8 muls, 4 adds)
//     for each output block C[i,j] (i,j in {0,1,2}):
//       sum over k in {0,1,2} of A[i,k] * B[k,j]  (3 sub-results added)
//     assemble 9 output blocks into 36-element 6x6 result

!val = i32
!svec = !secret.secret<tensor<?x!val>>

func.func @mm(
    %A  : !svec,
    %B  : !svec,
    %n  : !val { biscotti.progress_argument = 2 }
) -> !svec {

    %c2 = arith.constant 2 : !val
    %c3 = arith.constant 3 : !val

    %cond = arith.cmpi sle, %n, %c2 : !val
    cf.cond_br %cond, ^base, ^recursive { biscotti.base_condition = 0 }

^base:
    // Base case: 2x2 matmul.
    // A, B are tensor<4xi32> (row-major 2x2).
    // C[i,j] = sum_k A[i*2+k] * B[k*2+j] for i,j,k in {0,1}.
    %result_base = secret.generic(%A : !svec, %B : !svec) {
        ^bb0(%a: tensor<?x!val>, %b: tensor<?x!val>):
            %c0 = arith.constant 0 : index
            %c1_i = arith.constant 1 : index
            %c2_i = arith.constant 2 : index
            %c3_i = arith.constant 3 : index

            %a0 = tensor.extract %a[%c0]   : tensor<?x!val>
            %a1 = tensor.extract %a[%c1_i] : tensor<?x!val>
            %a2 = tensor.extract %a[%c2_i] : tensor<?x!val>
            %a3 = tensor.extract %a[%c3_i] : tensor<?x!val>

            %b0 = tensor.extract %b[%c0]   : tensor<?x!val>
            %b1 = tensor.extract %b[%c1_i] : tensor<?x!val>
            %b2 = tensor.extract %b[%c2_i] : tensor<?x!val>
            %b3 = tensor.extract %b[%c3_i] : tensor<?x!val>

            // C[0,0] = a0*b0 + a1*b2
            %p00_0 = arith.muli %a0, %b0 : !val
            %p00_1 = arith.muli %a1, %b2 : !val
            %c00 = arith.addi %p00_0, %p00_1 : !val
            // C[0,1] = a0*b1 + a1*b3
            %p01_0 = arith.muli %a0, %b1 : !val
            %p01_1 = arith.muli %a1, %b3 : !val
            %c01 = arith.addi %p01_0, %p01_1 : !val
            // C[1,0] = a2*b0 + a3*b2
            %p10_0 = arith.muli %a2, %b0 : !val
            %p10_1 = arith.muli %a3, %b2 : !val
            %c10 = arith.addi %p10_0, %p10_1 : !val
            // C[1,1] = a2*b1 + a3*b3
            %p11_0 = arith.muli %a2, %b1 : !val
            %p11_1 = arith.muli %a3, %b3 : !val
            %c11 = arith.addi %p11_0, %p11_1 : !val

            %out = tensor.from_elements %c00, %c01, %c10, %c11 : tensor<4x!val>
            %out_cast = tensor.cast %out : tensor<4x!val> to tensor<?x!val>
            secret.yield %out_cast : tensor<?x!val>
    } -> (!svec)
    return %result_base : !svec

^recursive:
    // 3-way split at n=6. Block size b = n/3 = 2. bsq = 4.
    %b = arith.divsi %n, %c3 : !val
    %bsq = arith.muli %b, %b : !val
    %b_idx = arith.index_cast %b : !val to index
    %bsq_idx = arith.index_cast %bsq : !val to index
    %n_idx = arith.index_cast %n : !val to index

    // Constants for block-row and block-col indices used during extraction.
    %br0 = arith.constant 0 : index
    %br1 = arith.constant 1 : index
    %br2 = arith.constant 2 : index

    // ---- Extract 9 A-blocks. For block (br, bc):
    //   flat_index = (br*b + local_row) * n + (bc*b + local_col)
    // where local_idx = local_row * b + local_col.

    // Helper macro-shape: one secret.generic per block. Repeated 9 times for A
    // and 9 times for B. All identical modulo the (br, bc) constants folded in.

    // ---- A blocks ----
    %a00 = secret.generic(%A : !svec, %br0 : index, %br0 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a01 = secret.generic(%A : !svec, %br0 : index, %br1 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a02 = secret.generic(%A : !svec, %br0 : index, %br2 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a10 = secret.generic(%A : !svec, %br1 : index, %br0 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a11 = secret.generic(%A : !svec, %br1 : index, %br1 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a12 = secret.generic(%A : !svec, %br1 : index, %br2 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a20 = secret.generic(%A : !svec, %br2 : index, %br0 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a21 = secret.generic(%A : !svec, %br2 : index, %br1 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %a22 = secret.generic(%A : !svec, %br2 : index, %br2 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%a: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %a[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    // ---- B blocks ----
    %b00 = secret.generic(%B : !svec, %br0 : index, %br0 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b01 = secret.generic(%B : !svec, %br0 : index, %br1 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b02 = secret.generic(%B : !svec, %br0 : index, %br2 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b10 = secret.generic(%B : !svec, %br1 : index, %br0 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b11 = secret.generic(%B : !svec, %br1 : index, %br1 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b12 = secret.generic(%B : !svec, %br1 : index, %br2 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b20 = secret.generic(%B : !svec, %br2 : index, %br0 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b21 = secret.generic(%B : !svec, %br2 : index, %br1 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    %b22 = secret.generic(%B : !svec, %br2 : index, %br2 : index, %b_idx : index, %n_idx : index, %bsq_idx : index) {
        ^bb0(%bv: tensor<?x!val>, %br: index, %bc: index, %bs: index, %nn: index, %bb: index):
            %out = tensor.generate %bb {
                ^bb0(%i: index):
                    %lr = arith.divui %i, %bs : index
                    %lc = arith.remui %i, %bs : index
                    %row_off = arith.muli %br, %bs : index
                    %col_off = arith.muli %bc, %bs : index
                    %row = arith.addi %row_off, %lr : index
                    %col = arith.addi %col_off, %lc : index
                    %src = arith.muli %row, %nn : index
                    %src1 = arith.addi %src, %col : index
                    %v = tensor.extract %bv[%src1] : tensor<?x!val>
                    tensor.yield %v : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    // 27 recursive calls with n=b (which is 2 for n=6).
    // For each output block C[i,j], accumulate A[i,k] * B[k,j] over k in {0,1,2}.
    // Indexing: t_{i*9 + j*3 + k}.

    // C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    %t0  = call @mm(%a00, %b00, %b) { biscotti.recursive_call = 0  } : (!svec, !svec, !val) -> !svec
    %t1  = call @mm(%a01, %b10, %b) { biscotti.recursive_call = 1  } : (!svec, !svec, !val) -> !svec
    %t2  = call @mm(%a02, %b20, %b) { biscotti.recursive_call = 2  } : (!svec, !svec, !val) -> !svec
    // C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    %t3  = call @mm(%a00, %b01, %b) { biscotti.recursive_call = 3  } : (!svec, !svec, !val) -> !svec
    %t4  = call @mm(%a01, %b11, %b) { biscotti.recursive_call = 4  } : (!svec, !svec, !val) -> !svec
    %t5  = call @mm(%a02, %b21, %b) { biscotti.recursive_call = 5  } : (!svec, !svec, !val) -> !svec
    // C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    %t6  = call @mm(%a00, %b02, %b) { biscotti.recursive_call = 6  } : (!svec, !svec, !val) -> !svec
    %t7  = call @mm(%a01, %b12, %b) { biscotti.recursive_call = 7  } : (!svec, !svec, !val) -> !svec
    %t8  = call @mm(%a02, %b22, %b) { biscotti.recursive_call = 8  } : (!svec, !svec, !val) -> !svec
    // C[1,0]
    %t9  = call @mm(%a10, %b00, %b) { biscotti.recursive_call = 9  } : (!svec, !svec, !val) -> !svec
    %t10 = call @mm(%a11, %b10, %b) { biscotti.recursive_call = 10 } : (!svec, !svec, !val) -> !svec
    %t11 = call @mm(%a12, %b20, %b) { biscotti.recursive_call = 11 } : (!svec, !svec, !val) -> !svec
    // C[1,1]
    %t12 = call @mm(%a10, %b01, %b) { biscotti.recursive_call = 12 } : (!svec, !svec, !val) -> !svec
    %t13 = call @mm(%a11, %b11, %b) { biscotti.recursive_call = 13 } : (!svec, !svec, !val) -> !svec
    %t14 = call @mm(%a12, %b21, %b) { biscotti.recursive_call = 14 } : (!svec, !svec, !val) -> !svec
    // C[1,2]
    %t15 = call @mm(%a10, %b02, %b) { biscotti.recursive_call = 15 } : (!svec, !svec, !val) -> !svec
    %t16 = call @mm(%a11, %b12, %b) { biscotti.recursive_call = 16 } : (!svec, !svec, !val) -> !svec
    %t17 = call @mm(%a12, %b22, %b) { biscotti.recursive_call = 17 } : (!svec, !svec, !val) -> !svec
    // C[2,0]
    %t18 = call @mm(%a20, %b00, %b) { biscotti.recursive_call = 18 } : (!svec, !svec, !val) -> !svec
    %t19 = call @mm(%a21, %b10, %b) { biscotti.recursive_call = 19 } : (!svec, !svec, !val) -> !svec
    %t20 = call @mm(%a22, %b20, %b) { biscotti.recursive_call = 20 } : (!svec, !svec, !val) -> !svec
    // C[2,1]
    %t21 = call @mm(%a20, %b01, %b) { biscotti.recursive_call = 21 } : (!svec, !svec, !val) -> !svec
    %t22 = call @mm(%a21, %b11, %b) { biscotti.recursive_call = 22 } : (!svec, !svec, !val) -> !svec
    %t23 = call @mm(%a22, %b21, %b) { biscotti.recursive_call = 23 } : (!svec, !svec, !val) -> !svec
    // C[2,2]
    %t24 = call @mm(%a20, %b02, %b) { biscotti.recursive_call = 24 } : (!svec, !svec, !val) -> !svec
    %t25 = call @mm(%a21, %b12, %b) { biscotti.recursive_call = 25 } : (!svec, !svec, !val) -> !svec
    %t26 = call @mm(%a22, %b22, %b) { biscotti.recursive_call = 26 } : (!svec, !svec, !val) -> !svec

    // 3-term sums: c_ij_block = t_{i*9+j*3} + t_{i*9+j*3+1} + t_{i*9+j*3+2}
    // For each output block, add three sub-results element-wise (2x2 -> 4 elems).

    %c00_blk = secret.generic(%t0 : !svec, %t1 : !svec, %t2 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c01_blk = secret.generic(%t3 : !svec, %t4 : !svec, %t5 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c02_blk = secret.generic(%t6 : !svec, %t7 : !svec, %t8 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c10_blk = secret.generic(%t9 : !svec, %t10 : !svec, %t11 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c11_blk = secret.generic(%t12 : !svec, %t13 : !svec, %t14 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c12_blk = secret.generic(%t15 : !svec, %t16 : !svec, %t17 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c20_blk = secret.generic(%t18 : !svec, %t19 : !svec, %t20 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c21_blk = secret.generic(%t21 : !svec, %t22 : !svec, %t23 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)
    %c22_blk = secret.generic(%t24 : !svec, %t25 : !svec, %t26 : !svec) {
        ^bb0(%x: tensor<?x!val>, %y: tensor<?x!val>, %z: tensor<?x!val>):
            %s0 = arith.addi %x, %y : tensor<?x!val>
            %s1 = arith.addi %s0, %z : tensor<?x!val>
            secret.yield %s1 : tensor<?x!val>
    } -> (!svec)

    // Assemble 9 blocks into 6x6 result.
    // For each output position i in [0, n*n):
    //   row = i / n, col = i % n
    //   block_row = row / b, block_col = col / b   (both in {0,1,2})
    //   local_row = row % b, local_col = col % b
    //   local_idx = local_row * b + local_col       (in [0, 4))
    //   pick c[block_row, block_col]_blk[local_idx]

    %nsq = arith.muli %n, %n : !val
    %nsq_idx = arith.index_cast %nsq : !val to index

    %result = secret.generic(
        %c00_blk : !svec, %c01_blk : !svec, %c02_blk : !svec,
        %c10_blk : !svec, %c11_blk : !svec, %c12_blk : !svec,
        %c20_blk : !svec, %c21_blk : !svec, %c22_blk : !svec,
        %b_idx : index, %n_idx : index, %nsq_idx : index
    ) {
        ^bb0(%q00: tensor<?x!val>, %q01: tensor<?x!val>, %q02: tensor<?x!val>,
             %q10: tensor<?x!val>, %q11: tensor<?x!val>, %q12: tensor<?x!val>,
             %q20: tensor<?x!val>, %q21: tensor<?x!val>, %q22: tensor<?x!val>,
             %bs: index, %nn: index, %nn2: index):
            %out = tensor.generate %nn2 {
                ^bb0(%i: index):
                    %row = arith.divui %i, %nn : index
                    %col = arith.remui %i, %nn : index
                    %br = arith.divui %row, %bs : index    // 0, 1, or 2
                    %bc = arith.divui %col, %bs : index    // 0, 1, or 2
                    %lr = arith.remui %row, %bs : index
                    %lc = arith.remui %col, %bs : index
                    %local_idx = arith.muli %lr, %bs : index
                    %local_idx1 = arith.addi %local_idx, %lc : index

                    // Extract from all 9 blocks; select the right one.
                    %v00 = tensor.extract %q00[%local_idx1] : tensor<?x!val>
                    %v01 = tensor.extract %q01[%local_idx1] : tensor<?x!val>
                    %v02 = tensor.extract %q02[%local_idx1] : tensor<?x!val>
                    %v10 = tensor.extract %q10[%local_idx1] : tensor<?x!val>
                    %v11 = tensor.extract %q11[%local_idx1] : tensor<?x!val>
                    %v12 = tensor.extract %q12[%local_idx1] : tensor<?x!val>
                    %v20 = tensor.extract %q20[%local_idx1] : tensor<?x!val>
                    %v21 = tensor.extract %q21[%local_idx1] : tensor<?x!val>
                    %v22 = tensor.extract %q22[%local_idx1] : tensor<?x!val>

                    %c0i = arith.constant 0 : index
                    %c1i = arith.constant 1 : index
                    %bc_is_0 = arith.cmpi eq, %bc, %c0i : index
                    %bc_is_1 = arith.cmpi eq, %bc, %c1i : index
                    %br_is_0 = arith.cmpi eq, %br, %c0i : index
                    %br_is_1 = arith.cmpi eq, %br, %c1i : index

                    // Row 0: pick among v00, v01, v02 by bc.
                    %r0_a = arith.select %bc_is_1, %v01, %v02 : !val
                    %r0   = arith.select %bc_is_0, %v00, %r0_a : !val
                    // Row 1: pick among v10, v11, v12 by bc.
                    %r1_a = arith.select %bc_is_1, %v11, %v12 : !val
                    %r1   = arith.select %bc_is_0, %v10, %r1_a : !val
                    // Row 2: pick among v20, v21, v22 by bc.
                    %r2_a = arith.select %bc_is_1, %v21, %v22 : !val
                    %r2   = arith.select %bc_is_0, %v20, %r2_a : !val
                    // Pick among r0, r1, r2 by br.
                    %rr_a = arith.select %br_is_1, %r1, %r2 : !val
                    %val  = arith.select %br_is_0, %r0, %rr_a : !val

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
