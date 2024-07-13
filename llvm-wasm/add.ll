; ModuleID = 'add_module'
source_filename = "add.ll"

define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}
