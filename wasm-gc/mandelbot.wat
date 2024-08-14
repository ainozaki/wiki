(module
  (type $complex (struct (field $re (mut f64)) (field $im (mut f64))))

  (func $calc_abs
    (param $c (ref null $complex))
    (result f64)
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

  (func $make
    (result (ref null $complex))
    (struct.new $complex (f64.const 1.0)(f64.const 2.0))
  )

  (func (export "test") (result f64)
    (local $c (ref null $complex))
    (local.set $c (call $make ))
    (call $calc_abs (local.get $c))
  )

  (func (export "ifMandelbrotIncluded")
    (param $re f64) (param $im f64)
    (result i32)
    (local $i i32)
    (local $sum_re f64)
    (local $sum_im f64)
    (local $tmp_re f64)
    (local $tmp_im f64)

    (local.set $sum_re (f64.const 0.0))
    (local.set $sum_im (f64.const 0.0))
    (local.set $i (i32.const 0))
    
    (loop $loop_divergence (result i32)
      local.get $sum_re
      local.set $tmp_re
      local.get $sum_im
      local.set $tmp_im

      ;; add one to $i
      local.get $i
      i32.const 1
      i32.add
      local.set $i

      ;; Calculate abs(tmp)
      local.get $sum_re
      local.get $sum_re
      f64.mul
      local.get $sum_im
      local.get $sum_im
      f64.mul
      f64.add
      
      ;; if abs(tmp) > 2.0, break the loop and return 0
      f64.const 4.0
      f64.gt
      (if
        (then
          i32.const 0
          return
        )
      )

      ;; sum = sum * sum + c;
      ;; re
      local.get $tmp_re
      local.get $tmp_re
      f64.mul
      local.get $tmp_im
      local.get $tmp_im
      f64.mul
      f64.sub
      local.get $re
      f64.add
      local.set $sum_re
      ;; im
      local.get $tmp_re
      local.get $tmp_im
      f64.mul
      f64.const 2.0
      f64.mul
      local.get $im
      f64.add
      local.set $sum_im

      ;; if $i < 20, branch to $loop_divergence
      local.get $i
      i32.const 20
      i32.lt_s
      br_if $loop_divergence

      ;; return 0
      i32.const 1
    )
  )
)
