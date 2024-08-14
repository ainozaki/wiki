(module
  (type $complex (struct (field $re (mut f64)) (field $im (mut f64))))

  (func $calc_abs
    (param $c (ref null $complex))
    (result f64)
    ;; local variables
    (local $re f64)
    (local $im f64)

    ;; access to ref $complex
    local.get $c
    struct.get $complex 0
    local.set $re
    local.get $c
    struct.get $complex 1
    local.set $im

    ;; Calculate abs
    local.get $re
    local.get $re
    f64.mul
    local.get $im
    local.get $im
    f64.mul
    f64.add
  )

  (func $update_complex
    (param $tmp (ref null $complex))
    (param $init (ref null $complex))
    ;; local variables
    (local $re f64)
    (local $im f64)

    ;; access to ref $complex
    local.get $tmp
    struct.get $complex 0
    local.set $re
    local.get $tmp
    struct.get $complex 1
    local.set $im

    ;; tmp = tmp * tmp + init
    ;; re
    local.get $re
    local.get $re
    f64.mul
    local.get $im
    local.get $im
    f64.mul
    f64.sub
    local.get $init
    struct.get $complex 0
    f64.add
    local.set $re
    ;; update tmp
    (struct.set $complex 0 (local.get $tmp) (local.get $re))

    ;; im
    local.get $re
    local.get $im
    f64.mul
    f64.const 2.0
    f64.mul
    local.get $init
    struct.get $complex 1
    f64.add
    local.set $im
    ;; update tmp
    (struct.set $complex 1 (local.get $tmp) (local.get $im))
  )

  (func $MandelbrotInternal 
    (param $init (ref null $complex)) (result i32) 
    ;; local variables
    (local $i i32)
    (local $tmp (ref null $complex))
    (local.set $i (i32.const 0))
    (local.set $tmp (call $make (f64.const 0.0) (f64.const 0.0)))

    ;; loop
    (loop $loop_divergence
      ;; add one to $i
      local.get $i
      i32.const 1
      i32.add
      local.set $i

      ;; Calculate abs(tmp)
      local.get $tmp
      call $calc_abs
      
      ;; if abs(tmp) > 2.0, break the loop and return 0
      f64.const 4.0
      f64.gt
      (if
        (then
          i32.const 0
          return
        )
      )

      ;; update tmp
      local.get $tmp
      local.get $init
      call $update_complex

      ;; if $i < 20, branch to $loop_divergence
      local.get $i
      i32.const 20
      i32.lt_s
      br_if $loop_divergence
    )
    ;; return 1
    i32.const 1
  )  

  (func $make
    (param $re f64) (param $im f64)
    (result (ref null $complex))
    (struct.new $complex (local.get $re) (local.get $im))
  )

  (func (export "Mandelbrot") (param $re f64) (param $im f64) (result i32)
    (local $c (ref null $complex))
    (local.set $c (call $make (local.get $re) (local.get $im)))
    (call $MandelbrotInternal (local.get $c))
  )
)
