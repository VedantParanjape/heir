#!/usr/bin/env python3
"""
Emit a recursive determinant MLIR file for an N x N matrix.

Cofactor (Laplace) expansion along row 0:

  det(A) = sum_{j=0}^{n-1} (-1)^j * A[0][j] * det(minor_j)

where minor_j is the (n-1) x (n-1) matrix obtained by deleting row 0 and
column j. Base case (n == 2): det = A[0][0]*A[1][1] - A[0][1]*A[1][0].

Recursion shape for biscotti:
  * @det(mat, n)  with `n` as the biscotti.progress_argument.
    `mat` is the flattened row-major n*n matrix (encrypted); the result is
    a length-1 tensor holding the scalar determinant.
  * The recursive branch ALWAYS emits MAX_ARITY tagged cofactor calls,
    each on an (n-1)x(n-1) minor with progress arg n-1. This keeps the arity
    uniform so the recursion pass (which clones one self-recursive function
    per node) builds a clean tree. For j >= n the term is masked to zero
    (coefficient 0) and its minor column index is clamped in-bounds, so the
    extra call is harmless. n=5 -> 5 real terms; n=4 -> 4 real + 1 masked; etc.

Intended targets are sizes 3..5 (MAX_ARITY = 5 covers them).

Usage:
  python3 gen_det.py 3                 # writes src/det_3.mlir
  python3 gen_det.py 5 -o path.mlir    # explicit output path
  python3 gen_det.py --all             # writes src/det_{3,4,5}.mlir
"""

import argparse
import os
import textwrap

DEFAULT_SIZES = [3, 4, 5]
MAX_ARITY = 5  # max cofactor terms emitted per level (covers N up to 5)


def emit_recursive_call(j: int) -> str:
  """The recursive call for cofactor term j. Emitted separately from the minor
  so that ALL minor generics come first and ALL @det calls are emitted
  consecutively (like matmul: every quadrant-extraction generic, then every
  call back-to-back). A clean run of adjacent sibling calls is what the
  recursion pass NW-merges; interleaving calls between the minor generics is
  what tangled the merge/reassembly before."""
  return textwrap.indent(
      f"%sub{j} = call @det(%minor{j}, %nm1) {{ biscotti.recursive_call ="
      f" {j} }} : (!svec, !val) -> !svec\n",
      "            ",
  )


def emit_minor(j: int, need_mask: bool) -> str:
  """MLIR for one cofactor minor j: build minor_j (drop row 0, column j)
  (12-space indented to sit inside ^recursive). Emits %minor{j}. The recursive
  %sub{j} is emitted separately by emit_recursive_call so all calls stay
  contiguous.

  This generic is PURE data movement -- it only extracts the minor's elements;
  no arithmetic. The cofactor coefficient a_j = A[0][j] is applied later in the
  reduction (t_j = a_j * sub_j), NOT here: a ciphertext*ciphertext multiply
  inside the extraction loop tangles the scheduler, so extraction stays a plain
  gather and all compute lives in the base case + the reduction."""
  del need_mask  # coefficient/mask now live in the reduction, not the minor
  body = textwrap.dedent(f"""\
    // ---- cofactor minor j={j}: drop row 0, column {j} (pure extraction) ----
    %jc{j}_idx = arith.constant {j} : index
    %minor{j} = secret.generic(%mat : !svec, %nm1_idx : index, %n_idx : index, %jc{j}_idx : index, %nm1sq_idx : index) {{
    ^bb0(%m{j}: tensor<?x!val>, %nm1b{j}: index, %nb{j}: index, %jcb{j}: index, %nm1sqb{j}: index):
      %out{j} = tensor.generate %nm1sqb{j} {{
      ^bb0(%i{j}: index):
        %rp{j} = arith.divsi %i{j}, %nm1b{j} : index                // minor row r'
        %cp{j} = arith.remsi %i{j}, %nm1b{j} : index                // minor col c'
        %rp1{j} = arith.addi %rp{j}, %one_idx : index               // r'+1 (skip row 0)
        %rowoff{j} = arith.muli %rp1{j}, %nb{j} : index             // (r'+1)*n
        %ge{j} = arith.cmpi sge, %cp{j}, %jcb{j} : index            // c' >= jc ?
        %gez{j} = arith.extui %ge{j} : i1 to i64                    // 0/1 (unsigned!)
        %skip{j} = arith.index_cast %gez{j} : i64 to index
        %cpp{j} = arith.addi %cp{j}, %skip{j} : index               // c' + (c'>=jc)
        %src{j} = arith.addi %rowoff{j}, %cpp{j} : index
        %v{j} = tensor.extract %m{j}[%src{j}] : tensor<?x!val>
        tensor.yield %v{j} : !val
      }} : tensor<?x!val>
      secret.yield %out{j} : tensor<?x!val>
    }} -> (!svec)
    """)
  return textwrap.indent(body, "            ")  # 12 spaces to sit in ^recursive


def emit_mlir(N: int) -> str:
  if N < 2:
    raise ValueError(f"N must be >= 2, got {N}")
  if N > MAX_ARITY:
    raise ValueError(f"N must be <= MAX_ARITY={MAX_ARITY}, got {N}")

  total = N * N
  # Exactly N cofactor terms (expansion along row 0 of an N x N). A single
  # shared @det template is specialized per recursion level, so det_N's template
  # has N calls; when it runs at a smaller n (det_4's n=3 children) the maskv_j
  # (j >= n) zeroes the extra term. For det_2/det_3 there is no smaller level,
  # so the masks are always 1 and simply fold away.
  arity = N
  # Masking (maskv_j) is only needed when the arity-N template can run at a
  # smaller n -- i.e. det_N with N >= 4 (its n=3 sub-levels). det_2/det_3 never
  # do, so we omit the masks there entirely (no dead 0/1 gates).
  need_mask = N >= 4
  # matmul layout: ALL minor generics first, then ALL recursive calls back-to-back.
  minors = "".join(emit_minor(j, need_mask) for j in range(arity))
  calls = "".join(emit_recursive_call(j) for j in range(arity))
  terms = minors + calls

  # --- In-range mask precompute (det_N, N>=4 only): maskv_j = (j < n) ? 1 : 0,
  # folded into the coefficient a_j below so masked terms contribute 0. ---
  mask_pre = "".join(f"""            %cj{j} = arith.constant {j} : !val
            %mask{j} = arith.cmpi slt, %cj{j}, %n : !val   // in-range gate (1 if j<n)
            %maskv{j} = arith.select %mask{j}, %one_val, %zero_val : !val
""" for j in range(arity)) if need_mask else ""

  # --- Coefficient generic: pull the row-0 elements a_j = mat[j] into N
  # length-1 secrets, with the cofactor SIGN folded in (odd j negated) and the
  # in-range mask folded in when need_mask. Folding the sign here makes the
  # reduction a pure ADDITIVE sum (t0 + t1 + t2 ...), which is what lowers to a
  # clean cross-lane rotate-reduction (like dot's uniform sum) -- an alternating
  # subi/addi reduction does not vectorize and gets left half-lowered. Pure
  # gather + sign; the ctxt*ctxt a_j*sub_j multiply happens in the reduction. ---
  coeff_operand = (
      "".join(f", %maskv{j} : !val" for j in range(arity)) if need_mask else ""
  )
  coeff_blockarg = (
      "".join(f", %mvb{j}: !val" for j in range(arity)) if need_mask else ""
  )

  def coeff_lines(j):
    # a_j = A[0][j] = mat[j], a RAW matrix load (masked when need_mask). The
    # cofactor SIGN is NOT folded here -- doing so would make odd a_j a computed
    # value (-mat[j]) that Coyote's input packing can't lay into a lane. The sign
    # lives in the reduction's alternating subi/addi instead, so every coeff
    # stays a raw load that packs cleanly.
    lines = [
        f"                %ce{j} = arith.constant {j} : index",
        f"                %cv{j} = tensor.extract %cm[%ce{j}] : tensor<?x!val>",
    ]
    val = f"%cv{j}"
    if need_mask:
      lines.append(
          f"                %cvm{j} = arith.muli {val}, %mvb{j} : !val   //"
          " in-range mask"
      )
      val = f"%cvm{j}"
    return "\n".join(lines), val  # yield the SCALAR i32 coefficient directly

  coeff_pairs = [coeff_lines(j) for j in range(arity)]
  coeff_body = "\n".join(p[0] for p in coeff_pairs)
  a_res = ", ".join(f"%a{j}" for j in range(arity))
  coeff_yield = ", ".join(p[1] for p in coeff_pairs)
  det_types = ", ".join("!val" for _ in range(arity))  # scalar i32 coeffs
  det_results = ", ".join("!cff" for _ in range(arity))  # !secret.secret<i32>

  # --- ONE reduction: det = t0 + t1 + t2 + ... , t_j = a_j * sub_j (a_j carries
  # the cofactor sign). The coeffs a_j are scalar i32 secrets -- so as reduction
  # block args they are `i32`, which wrapBlockArgsWithVirtualLoads stubs with
  # __coyote_load (and that stub survives the merge). The subs stay !svec tensors
  # and are extracted to scalars; muli/addi are scalar, then re-wrapped. ---
  a_ops = ", ".join(f"%a{j} : !cff" for j in range(arity))
  sub_ops = ", ".join(f"%sub{j} : !svec" for j in range(arity))
  aa_args = ", ".join(f"%aa{j}: !val" for j in range(arity))
  s_args = ", ".join(f"%s{j}: tensor<?x!val>" for j in range(arity))
  term_lines = "\n".join(
      f"                %sv{j} = tensor.extract %s{j}[%rzero] :"
      f" tensor<?x!val>\n                %ad{j} = arith.muli %aa{j}, %sv{j} :"
      " !val"
      for j in range(arity)
  )

  # Alternating cofactor sum over the scalar terms: acc = ad0; acc = acc -/+ ad_j
  # (subi if j odd, addi if j even). The sign is HERE, not in the coeffs, so a_j
  # stay raw loads. Coyote schedules the mixed subi/addi in the reduction.
  sum_parts = []
  prev = "%ad0"
  for j in range(1, arity):
    op = "subi" if (j % 2 == 1) else "addi"
    name = "%dets" if j == arity - 1 else f"%acc{j}"
    sum_parts.append(
        f"                {name} = arith.{op} {prev}, %ad{j} : !val"
    )
    prev = name
  if arity == 1:  # not used for det (N>=2), but keep it total
    sum_parts.append("                %dets = arith.addi %ad0, %ad0 : !val")
  sum_code = "\n".join(sum_parts)

  # Human-readable sign string for the comment: "+t0 -t1 +t2 ...".
  sign_str = " ".join(
      ("+" if j % 2 == 0 else "-") + f"t{j}" for j in range(arity)
  )

  return textwrap.dedent(f"""\
        // Recursive determinant of an N={N} matrix via cofactor expansion
        // along row 0. Input: flattened row-major N*N = {total} matrix
        // (encrypted). Output: length-1 tensor holding det.
        //
        // Generated by benchmarks/det/gen_det.py -- do not hand-edit.

        !val = i32
        !svec = !secret.secret<tensor<?x!val>>   // one dynamic type: matrix, minor, and det result
        !cff  = !secret.secret<!val>             // scalar cofactor coefficient

        func.func @det(
            %mat : !svec,
            %n   : !val {{ biscotti.progress_argument = 1 }}
        ) -> !svec {{
            %c2v      = arith.constant 2 : !val
            %one_val  = arith.constant 1 : !val
            %zero_val = arith.constant 0 : !val

            %cond = arith.cmpi sle, %n, %c2v : !val
            cf.cond_br %cond, ^base, ^recursive {{ biscotti.base_condition = 0 }}

        ^base:
            // n == 2: det = m[0]*m[3] - m[1]*m[2]  (row-major 2x2).
            %rb = secret.generic(%mat : !svec) {{
                ^bb0(%m: tensor<?x!val>):
                    %i0 = arith.constant 0 : index
                    %i1 = arith.constant 1 : index
                    %i2 = arith.constant 2 : index
                    %i3 = arith.constant 3 : index
                    %a = tensor.extract %m[%i0] : tensor<?x!val>
                    %b = tensor.extract %m[%i1] : tensor<?x!val>
                    %c = tensor.extract %m[%i2] : tensor<?x!val>
                    %d = tensor.extract %m[%i3] : tensor<?x!val>
                    %ad = arith.muli %a, %d : !val
                    %bc = arith.muli %b, %c : !val
                    %det = arith.subi %ad, %bc : !val
                    %out = tensor.from_elements %det : tensor<1x!val>
                    %out_cast = tensor.cast %out : tensor<1x!val> to tensor<?x!val>
                    secret.yield %out_cast : tensor<?x!val>
            }} -> (!svec)
            return %rb : !svec

        ^recursive:
            // n > 2: cofactor expansion along row 0 -> N minors, each recursed.
            // det = sum_j (-1)^j * a_j * det(minor_j), a_j = A[0][j]. Minors are
            // pure extraction; a_j*sub_j is done in the reduction.
            %one_idx  = arith.constant 1 : index
            %nm1      = arith.subi %n, %one_val : !val
            %nm1_idx  = arith.index_cast %nm1 : !val to index
            %n_idx    = arith.index_cast %n : !val to index
            %nm1sq    = arith.muli %nm1, %nm1 : !val
            %nm1sq_idx = arith.index_cast %nm1sq : !val to index

{mask_pre}            // Cofactor coefficients a_j = A[0][j] = mat[j] (masked when need_mask).
            {a_res} = secret.generic(%mat : !svec{coeff_operand}) {{
            ^bb0(%cm: tensor<?x!val>{coeff_blockarg}):
{coeff_body}
                secret.yield {coeff_yield} : {det_types}
            }} -> ({det_results})

{terms}
            // Single reduction after ALL recursive calls: t_j = a_j * sub_j, then
            // the alternating cofactor sum (subi odd, addi even). Coeffs are raw.
            //   det = {sign_str}
            %result = secret.generic({a_ops}, {sub_ops}) {{
            ^bb0({aa_args}, {s_args}):
                %rzero = arith.constant 0 : index
{term_lines}
{sum_code}
                %dett = tensor.from_elements %dets : tensor<1x!val>
                %detv = tensor.cast %dett : tensor<1x!val> to tensor<?x!val>
                secret.yield %detv : tensor<?x!val>
            }} -> (!svec)
            return %result : !svec
        }}

        // Entry: flattened row-major N*N matrix -> scalar det (static length-1).
        !inputvec  = !secret.secret<tensor<{total}x!val>>
        !outputvec = !secret.secret<tensor<1x!val>>

        func.func @main(%flat : !inputvec) -> !outputvec {{
            %ck = arith.constant {N} : !val

            %mat_dyn = secret.generic(%flat : !inputvec) {{
            ^bb0(%f: tensor<{total}x!val>):
                %cast = tensor.cast %f : tensor<{total}x!val> to tensor<?x!val>
                secret.yield %cast : tensor<?x!val>
            }} -> !svec

            %det_dyn = call @det(%mat_dyn, %ck)
                {{ biscotti.call = 0 }} : (!svec, !val) -> !svec

            // Cast the dynamic det result back to a static length-1 output
            // (matches dot: @det returns !svec dynamic, main returns static).
            %det_val = secret.generic(%det_dyn : !svec) {{
            ^bb0(%r: tensor<?x!val>):
                %cast = tensor.cast %r : tensor<?x!val> to tensor<1x!val>
                secret.yield %cast : tensor<1x!val>
            }} -> !outputvec

            return %det_val : !outputvec
        }}
        """)


def main():
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("N", nargs="?", type=int, help="matrix dimension (2..4)")
  ap.add_argument(
      "-o", "--output", help="output path (default: src/det_<N>.mlir)"
  )
  ap.add_argument(
      "--all",
      action="store_true",
      help=f"emit all default sizes: {DEFAULT_SIZES}",
  )
  args = ap.parse_args()

  here = os.path.dirname(os.path.abspath(__file__))
  src_dir = os.path.join(here, "src")
  os.makedirs(src_dir, exist_ok=True)

  if args.all:
    for n in DEFAULT_SIZES:
      path = os.path.join(src_dir, f"det_{n}.mlir")
      with open(path, "w") as f:
        f.write(emit_mlir(n))
      print(f"wrote {path}")
    return

  if args.N is None:
    ap.error("give a size N (or --all)")
  path = args.output or os.path.join(src_dir, f"det_{args.N}.mlir")
  with open(path, "w") as f:
    f.write(emit_mlir(args.N))
  print(f"wrote {path}")


if __name__ == "__main__":
  main()
