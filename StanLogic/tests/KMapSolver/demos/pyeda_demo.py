from pyeda.inter import *

a, b, c = map(exprvar, 'abc')

def minterm_expr(m, vars):
    return And(*[
        var if (m >> i) & 1 else ~var
        for i, var in enumerate(reversed(vars))
    ])

vars = [a, b, c]

on_set = [1, 3, 7]
dc_set = [5]

f_on = Or(*[minterm_expr(m, vars) for m in on_set])
f_dc = Or(*[minterm_expr(m, vars) for m in dc_set])

f_sop = espresso_exprs(f_on, f_dc)[0]
print(f_sop)
