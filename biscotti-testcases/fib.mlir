func.func @fib(%n : i32 { biscotti.progress_argument = 0 }) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    // Base cases
    %cond = arith.cmpi slt, %n, %c2 : i32
    cf.cond_br %cond, ^base, ^recursive { biscotti.base_condition = 0 }
    
^base:
    return %n : i32
    
^recursive:
    // Calculate fib(n-1)
    %n_minus_1 = arith.subi %n, %c1 : i32
    %fib_n_minus_1 = call @fib(%n_minus_1) { biscotti.recursive_call = 0 } : (i32) -> i32
    
    // Calculate fib(n-2)
    %n_minus_2 = arith.subi %n, %c2 : i32
    %fib_n_minus_2 = call @fib(%n_minus_2) { biscotti.recursive_call = 1 }: (i32) -> i32

    // Sum the results
    %result = arith.addi %fib_n_minus_1, %fib_n_minus_2 : i32
    return %result : i32
}

func.func @main() -> i32 {
    %n = arith.constant 6 : i32
    %result = call @fib(%n) { biscotti.call = 0 } : (i32) -> i32 
    return %result : i32
}

