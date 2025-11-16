'''
Demo  program for Ones Complement addition.

Author: Somtochukwu Emeka-Onwuneme
Institution: Stan's Technologies
'''

from stanlogic import OnesComplement

oc = OnesComplement(bit_length = 4)

oc.set_decimals([5,-3,2])

convert = oc.decimal_to_binary()
print("Binary form:", convert)

total = oc.add_binaries()
print("1's complement addition:", total)