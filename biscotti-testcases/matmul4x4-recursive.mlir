!val = i32
!svec = !secret.secret<tensor<?x!val>>

func.func @mm(
    %A  : !svec,
    %B  : !svec,
    %n  : !val { biscotti.progress_argument = 2 }
) -> !svec {

    %c1 = arith.constant 1 : !val
    %c2 = arith.constant 2 : !val

    %cond = arith.cmpi sle, %n, %c1 : !val
    cf.cond_br %cond, ^base, ^recursive { biscotti.base_condition = 0 }

^base:
    // A and B are each tensor<1xi32>, just multiply
    %result_base = secret.generic(%A : !svec, %B : !svec) {
        ^bb0(%a: tensor<?x!val>, %b: tensor<?x!val>):
            %c0 = arith.constant 0 : index
            %va = tensor.extract %a[%c0] : tensor<?x!val>
            %vb = tensor.extract %b[%c0] : tensor<?x!val>
            %prod = arith.muli %va, %vb : !val
            %out = tensor.from_elements %prod : tensor<1x!val>
            %out_cast = tensor.cast %out : tensor<1x!val> to tensor<?x!val>
            secret.yield %out_cast : tensor<?x!val>
    } -> (!svec)
    return %result_base : !svec

^recursive:
    %half = arith.divsi %n, %c2 : !val
    %hsq = arith.muli %half, %half : !val
    %half_idx = arith.index_cast %half : !val to index
    %hsq_idx = arith.index_cast %hsq : !val to index
    %n_idx = arith.index_cast %n : !val to index

    // Extract quadrants: A is n*n flat, quadrant is half*half flat
    // A00 starts at row 0 col 0, A01 at row 0 col half, etc.
    // We need to extract non-contiguous elements, so we do it inside secret.generic

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

    // Same for B
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

    // 8 recursive calls on (half*half)-sized tensors
    %t0 = call @mm(%a00, %b00, %half) { biscotti.recursive_call = 0 } : (!svec, !svec, !val) -> !svec
    %t1 = call @mm(%a01, %b10, %half) { biscotti.recursive_call = 1 } : (!svec, !svec, !val) -> !svec
    %t2 = call @mm(%a00, %b01, %half) { biscotti.recursive_call = 2 } : (!svec, !svec, !val) -> !svec
    %t3 = call @mm(%a01, %b11, %half) { biscotti.recursive_call = 3 } : (!svec, !svec, !val) -> !svec
    %t4 = call @mm(%a10, %b00, %half) { biscotti.recursive_call = 4 } : (!svec, !svec, !val) -> !svec
    %t5 = call @mm(%a11, %b10, %half) { biscotti.recursive_call = 5 } : (!svec, !svec, !val) -> !svec
    %t6 = call @mm(%a10, %b01, %half) { biscotti.recursive_call = 6 } : (!svec, !svec, !val) -> !svec
    %t7 = call @mm(%a11, %b11, %half) { biscotti.recursive_call = 7 } : (!svec, !svec, !val) -> !svec

    // Pairwise add (each is half*half sized)
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

    // Assemble four (half*half) quadrants into (n*n) result
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
                    %row_half = arith.divui %row, %h : index
                    %col_half = arith.divui %col, %h : index
                    %row_local = arith.remui %row, %h : index
                    %col_local = arith.remui %col, %h : index
                    %local_idx = arith.muli %row_local, %h : index
                    %local_idx1 = arith.addi %local_idx, %col_local : index

                    // Select quadrant based on (row_half, col_half)
                    %v00 = tensor.extract %q00[%local_idx1] : tensor<?x!val>
                    %v01 = tensor.extract %q01[%local_idx1] : tensor<?x!val>
                    %v10 = tensor.extract %q10[%local_idx1] : tensor<?x!val>
                    %v11 = tensor.extract %q11[%local_idx1] : tensor<?x!val>

                    %c0 = arith.constant 0 : index
                    %c1_idx = arith.constant 1 : index
                    %is_top = arith.cmpi eq, %row_half, %c0 : index
                    %is_left = arith.cmpi eq, %col_half, %c0 : index

                    %top_val = arith.select %is_left, %v00, %v01 : !val
                    %bot_val = arith.select %is_left, %v10, %v11 : !val
                    %val = arith.select %is_top, %top_val, %bot_val : !val

                    tensor.yield %val : !val
            } : tensor<?x!val>
            secret.yield %out : tensor<?x!val>
    } -> (!svec)

    return %result : !svec
}

!inputvec = !secret.secret<tensor<16xi32>>
func.func @main(%A : !inputvec, %B : !inputvec) -> !inputvec {
    %c4 = arith.constant 4 : !val

    %A_dyn = secret.generic(%A : !inputvec) {
    ^bb0(%a: tensor<16xi32>):
        %cast = tensor.cast %a : tensor<16xi32> to tensor<?xi32>
        secret.yield %cast : tensor<?xi32>
    } -> !svec

    %B_dyn = secret.generic(%B : !inputvec) {
    ^bb0(%b: tensor<16xi32>):
        %cast = tensor.cast %b : tensor<16xi32> to tensor<?xi32>
        secret.yield %cast : tensor<?xi32>
    } -> !svec

    %result = call @mm(%A_dyn, %B_dyn, %c4)
        { biscotti.call = 0 } : (!svec, !svec, !val) -> !svec

    %result_static = secret.generic(%result : !svec) {
    ^bb0(%r: tensor<?xi32>):
        %cast = tensor.cast %r : tensor<?xi32> to tensor<16xi32>
        secret.yield %cast : tensor<16xi32>
    } -> !inputvec

    return %result_static : !inputvec
}
