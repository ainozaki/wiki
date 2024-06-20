# Build with `-pg`

```
cc -o myprog myprog.o utils.o -pg
```

# Analyze
```
apt install graphviz
pip install gprof2dot
gprof path/to/your/executable | gprof2dot.py | dot -Tpng -o output.png
```
