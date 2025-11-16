class OnesComplement:
    def __init__(self, bit_length):
        self.decimals = []      # list of input decimals
        self.binaries = []      # list of binary (strings)
        self.bit_length = bit_length

    def set_decimals(self, values):
        """Accept a list of decimal integers"""
        if not isinstance(values, list):
            raise TypeError("Input must be a list of integers.")
        self.decimals = [int(v) for v in values]

    def decimal_to_binary(self):
        """Convert decimals to fixed-width binary (1's complement for negatives)."""
        self.binaries = []

        for dec in self.decimals:
            # Create a list to hold bits
            bits = []

            # If the number is negative
            if dec < 0:
                # Get binary of absolute value
                val = abs(dec)

                # Convert to binary
                while val > 0:
                    # Get bits for absolute value, parse it into a list of strings
                    # modulo operator returns remainder
                    bits.append(str(val % 2))
                    
                    # Integer division to reduce value
                    val //= 2

                # Reverse bits to get correct order after collecting LSB to MSB    
                bits.reverse()

                # Form binary string and pad with leading zeros
                binary_form = "".join(bits) if bits else "0"

                # Pad to fixed bit length determined at initialization
                binary_form = binary_form.zfill(self.bit_length)

                # 1's complement, take the complement of each bit in the binary_form
                ones_complement = "".join("1" if b == "0" else "0" for b in binary_form)

                # Append to binaries list
                self.binaries.append(ones_complement)

            else:  # positive or zero
                while dec > 0:
                    bits.append(str(dec % 2))
                    dec //= 2
                bits.reverse()
                binary_form = "".join(bits) if bits else "0"
                binary_form = binary_form.zfill(self.bit_length)
                self.binaries.append(binary_form)

        return self.binaries

    def add_binaries(self):
        """Perform 1's complement addition of the binary numbers."""
        if not self.binaries:
            self.decimal_to_binary()

        # Find maximum length of binaries for alignment
        max_len = max(len(s) for s in self.binaries)
        carry = 0

        # Start with the first binary number
        initial = list(self.binaries[0])

        # Add each subsequent binary number
        for l in range(1, len(self.binaries)):
            carry = 0
            result = []

            # Add from right to left
            for j in range(max_len - 1, -1, -1):

                # Add the initial bit, current binary bit, and carry
                s = int(initial[j]) + int(self.binaries[l][j]) + carry
                result.insert(0, str(s % 2))

                # Use floor division to return carry for next bit
                carry = s // 2

            # End-around carry
            while carry:
                for j in range(len(result) - 1, -1, -1):
                    # Add carry to the least significant bit
                    s = int(result[j]) + carry

                    # Update the bit and carry
                    result[j] = str(s % 2)
                    # Perform floor division to get new carry
                    carry = s // 2

                    # If no carry, break out of loop
                    if carry == 0:
                        break

            initial = result

        return "".join(result)


# --- Separate demo program ---
def main():
    oc = OnesComplement(bit_length=4)

    # User provides list of integers
    oc.set_decimals([5, -3, 2])

    # Convert to binaries
    print("Binary forms:", oc.decimal_to_binary())

    # Perform 1's complement addition
    print("Final sum (1's complement):", oc.add_binaries())


if __name__ == "__main__":
    main()