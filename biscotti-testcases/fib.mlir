!val = i32
!ctxt = !secret.secret<!val>

func.func @fib(%i_cipher : !ctxt, %n : !val { biscotti.progress_argument = 1 }) -> !ctxt {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // Base cases
    %cond = arith.cmpi slt, %n, %c0 : i32
    cf.cond_br %cond, ^base, ^recursive { biscotti.base_condition = 0 }
    
^base:
    return %i_cipher : !ctxt
    
^recursive:
    // Calculate fib(n-1)
    %c1_cipher = secret.conceal %c1 : !val -> !ctxt
    %i_cipher_minus_1 = secret.generic(%i_cipher: !ctxt, %c1_cipher: !ctxt) {
        ^bb0(%X: !val, %Y: !val):
            %0 = arith.subi %X, %Y : !val
            secret.yield %0 : !val
    } -> (!ctxt)
    %n_minus_1 = arith.subi %n, %c1 : i32
    %fib_n_minus_1 = call @fib(%i_cipher_minus_1, %n_minus_1) { biscotti.recursive_call = 0 } : (!ctxt, !val) -> !ctxt
    
    // Calculate fib(n-2)
    %c2_cipher = secret.conceal %c2 : !val -> !ctxt
    %i_cipher_minus_2 = secret.generic(%i_cipher: !ctxt, %c2_cipher: !ctxt) {
        ^bb0(%X: !val, %Y: !val):
            %0 = arith.subi %X, %Y : !val
            secret.yield %0 : !val
    } -> (!ctxt)
    %n_minus_2 = arith.subi %n, %c2 : i32
    %fib_n_minus_2 = call @fib(%i_cipher_minus_2, %n_minus_2) { biscotti.recursive_call = 1 }: (!ctxt, !val) -> !ctxt

    // Sum the results
    %result = secret.generic(%fib_n_minus_1: !ctxt, %fib_n_minus_2: !ctxt) {
        ^bb0(%X: !val, %Y: !val):
            %0 = arith.addi %X, %Y : !val
            secret.yield %0 : !val
    } -> (!ctxt)
    return %result : !ctxt
}

func.func @main(%input_fib: !ctxt) -> !ctxt {
    %n = arith.constant 6 : i32
    %result = call @fib(%input_fib, %n) { biscotti.call = 0 } : (!ctxt, !val) -> !ctxt 
    // %n2 = arith.constant 10 : i32
    // %result2 = call @fib(%input_fib, %n2) { biscotti.call = 0 } : (!ctxt, !val) -> !ctxt
    return %result : !ctxt
}