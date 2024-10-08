(module
  (type $vec (array f32))
  (type $mvec (array (mut f32)))

  (global (ref $vec) (array.new $vec (f32.const 1) (i32.const 3)))
  (global (ref $vec) (array.new_default $vec (i32.const 3)))

  (func $new (export "new") (result (ref $vec))
    (array.new_default $vec (i32.const 3))
  )

  (func $get (param $i i32) (param $v (ref $vec)) (result f32)
    (array.get $vec (local.get $v) (local.get $i))
  )
  (func (export "get") (param $i i32) (result f32)
    (call $get (local.get $i) (call $new))
  )

  (func $set_get (param $i i32) (param $v (ref $mvec)) (param $y f32) (result f32)
    (array.set $mvec (local.get $v) (local.get $i) (local.get $y))
    (array.get $mvec (local.get $v) (local.get $i))
  )
  (func (export "set_get") (param $i i32) (param $y f32) (result f32)
    (call $set_get (local.get $i)
      (array.new_default $mvec (i32.const 3))
      (local.get $y)
    )
  )

  (func $len (param $v (ref array)) (result i32)
    (array.len (local.get $v))
  )
  (func (export "len") (result i32)
    (call $len (call $new))
  )
)