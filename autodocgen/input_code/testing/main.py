from math_utils import add_numbers, multiply_numbers
from string_utils import reverse_string

def main():
    result1 = add_numbers(5, 10)
    result2 = multiply_numbers(3, 4)
    result3 = reverse_string("Hello World")

    print("Addition Result:", result1)
    print("Multiplication Result:", result2)
    print("Reversed String:", result3)

if __name__ == "__main__":
    main()
