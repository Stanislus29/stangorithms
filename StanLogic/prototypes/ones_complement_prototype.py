print("Define number of binary integers to add")
num = input()
num_integers = int(num)

# create empty list to store binary integers
x = []

for i in range(num_integers):
    value = input("Enter binary integer: ")
    if all(bit in "01" for bit in value):   # check if valid binary
        x.append(value)
    else:
        print("Invalid binary number! Try again.")
        break

print("You entered:", x)

# print maximum length after the loop
print("Maximum length of entered binaries:", max(len(s) for s in x))

#Normalize binaries 
max_len = max(len(s) for s in x) #find maximum length in the binary list

# Pad each binary with zeros on the left
for l in range(len(x)):
    if len(x[l]) < max_len:
        x[l] = x[l].zfill(max_len) #adds 0 to the left of the binary number to be normalized until it satisfies the earlier condition

print("Normalized binaries:", x)

#Addition 
carry = 0 
initial = list(x[0])

for l in range(1, len(x)):
    carry = 0
    result = []

    for j in range(max_len -1, -1, -1):
        s = int(initial[j]) + int(x[l][j]) + carry 
        result.insert(0, str(s%2)) #prints remainder of addition
        carry = s//2 #prints quotient

    #End-around carry
    while carry:
        for j in range (len(result) -1, -1, -1):
            s = int(result[j]) + carry
            result[j] = str(s % 2) #Store the quotient in the position of j being calculated
            carry = s // 2
            if carry == 0:
                break

    result = initial

print("Final sum (1's complement):", "".join(result))
 