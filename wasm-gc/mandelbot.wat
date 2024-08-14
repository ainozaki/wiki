(module
  (func (export "ifMandelbrotIncluded")
    (param $re f32) (param $im f32)
    (result i32)
    (local $i i32)
    (local $tmp_re f32)
    (local $tmp_im f32)
    
    (local.get $re)
    (local.get $im)
    drop
    drop
    (loop $loop_divergence (result i32)
      ;; add one to $i
      local.get $i
      i32.const 1
      i32.add
      local.set $i

      ;; Calculate abs(tmp)
      local.get $tmp_re
      local.get $tmp_re
      f32.mul
      local.get $tmp_im
      local.get $tmp_im
      f32.mul
      f32.add
      
      ;; if abs(tmp) > 2.0, break the loop and return 1
      f32.const 4.0
      f32.gt
      (if
        (then
          i32.const 1
          return
        )
      )

      ;; if $i < 20, branch to $loop_divergence
      local.get $i
      i32.const 20
      i32.lt_s
      br_if $loop_divergence

      ;; return 0
      i32.const 0
    )
  )
)
