import random

def generate_quadratic_samples(num_samples=5):
    samples = []
    for _ in range(num_samples):
        # Random coefficients for quadratic function: a, b, c (0 to 3)
        a, b, c = random.randint(0, 5), random.randint(0, 5), random.randint(0, 5)
        
        # Random x value between 0 and 9
        x = random.randint(0, 9)

        # Compute quadratic function result: y = ax^2 + bx + c (mod 11)
        y = (a * (x ** 2) + b * x + c) 

        # Create one-hot encoded output matrix
        if not y >= 10:
            output_matrix = [0] * 11
            output_matrix[y] = 1  # Set the correct index to 1

            samples.append(f"{a, b, c, x}{output_matrix}")

    return samples

with open("Data.txt", "w") as f:
    for sample in generate_quadratic_samples(100):
        l = ""
        for j in sample:
            if j.isdigit():
                l+=j
            print(j, end="")
        f.write(l + "\n")
