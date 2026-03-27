!val = i32
!vec = tensor<16x!val>
!secret_vec = !secret.secret<!vec>

func.func @mm(
    %C  : !secret_vec,
    %A  : !secret_vec,
    %B  : !secret_vec,
    %w  : !val,
    %n  : !val { biscotti.progress_argument = 0 },
    %sa : !val,
    %sb : !val
) -> !secret_vec {

    %c1 = arith.constant 1 : !val
    %c2 = arith.constant 2 : !val

    %cond = arith.cmpi eq, %n, %c1 : !val
    cf.cond_br %cond, ^base, ^recursive { biscotti.base_condition = 0 }

^base:
    %row = arith.divsi %sa, %w : !val
    %col = arith.remsi %sb, %w : !val
    %out_val = arith.muli %row, %w : !val
    %out_idx = arith.addi %out_val, %col : !val

    %result_base = secret.generic(
        %C : !secret_vec, %A : !secret_vec, %B : !secret_vec,
        %sa : !val, %sb : !val, %out_idx : !val
    ) {
        ^bb0(%c: !vec, %a: !vec, %b: !vec, %ia: !val, %ib: !val, %oi: !val):
            %idx_a = arith.index_cast %ia : !val to index
            %idx_b = arith.index_cast %ib : !val to index
            %idx_o = arith.index_cast %oi : !val to index
            %va = tensor.extract %a[%idx_a] : !vec
            %vb = tensor.extract %b[%idx_b] : !vec
            %prod = arith.muli %va, %vb : !val
            %old = tensor.extract %c[%idx_o] : !vec
            %sum = arith.addi %old, %prod : !val
            %out = tensor.insert %sum into %c[%idx_o] : !vec
            secret.yield %out : !vec
    } -> (!secret_vec)
    return %result_base : !secret_vec

^recursive:
    %half = arith.divsi %n, %c2 : !val
    %half_w = arith.muli %half, %w : !val

    %sa01 = arith.addi %sa, %half : !val
    %sa10 = arith.addi %sa, %half_w : !val
    %sa11 = arith.addi %sa10, %half : !val

    %sb01 = arith.addi %sb, %half : !val
    %sb10 = arith.addi %sb, %half_w : !val
    %sb11 = arith.addi %sb10, %half : !val

    // C00 += A00*B00
    %c0 = call @mm(%C, %A, %B, %w, %half, %sa, %sb)
        { biscotti.recursive_call = 0 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec
    // C00 += A01*B10
    %c1r = call @mm(%c0, %A, %B, %w, %half, %sa01, %sb10)
        { biscotti.recursive_call = 1 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec

    // C01 += A00*B01
    %c2r = call @mm(%c1r, %A, %B, %w, %half, %sa, %sb01)
        { biscotti.recursive_call = 2 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec
    // C01 += A01*B11
    %c3r = call @mm(%c2r, %A, %B, %w, %half, %sa01, %sb11)
        { biscotti.recursive_call = 3 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec

    // C10 += A10*B00
    %c4r = call @mm(%c3r, %A, %B, %w, %half, %sa10, %sb)
        { biscotti.recursive_call = 4 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec
    // C10 += A11*B10
    %c5r = call @mm(%c4r, %A, %B, %w, %half, %sa11, %sb10)
        { biscotti.recursive_call = 5 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec

    // C11 += A10*B01
    %c6r = call @mm(%c5r, %A, %B, %w, %half, %sa10, %sb01)
        { biscotti.recursive_call = 6 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec
    // C11 += A11*B11
    %c7r = call @mm(%c6r, %A, %B, %w, %half, %sa11, %sb11)
        { biscotti.recursive_call = 7 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec

    return %c7r : !secret_vec
}

func.func @main(%A : !secret_vec, %B : !secret_vec) -> !secret_vec {
    %c0 = arith.constant 0 : !val
    %c4 = arith.constant 4 : !val
    %zero = arith.constant 0 : !val
    %C_plain = tensor.splat %zero : !vec
    %C = secret.conceal %C_plain : !vec -> !secret_vec
    %result = call @mm(%C, %A, %B, %c4, %c4, %c0, %c0)
        { biscotti.call = 0 }
        : (!secret_vec, !secret_vec, !secret_vec, !val, !val, !val, !val) -> !secret_vec
    return %result : !secret_vec
}
