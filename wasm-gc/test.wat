(module
  (type $boxed-i32 (struct (field i32)))

  (func (export "make")
    (param $i i32)
    (result (ref $boxed-i32))
    (struct.new $boxed-i32 (local.get $i))
  )
  (func (export "get")
    (param $o (ref $boxed-i32))
    (result i32)
    (struct.get $boxed-i32 0 (local.get $o))
  )
)
