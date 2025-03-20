import random

def generate_xor_samples(samples_per_class=1000):
    samples = []
    numbers = [0] * 2  # Track samples per class
    
    # For each target y value (0 to 3)
    for target_y in range(2):
        while numbers[target_y] < samples_per_class:
            # Random x and y
            x = random.randint(0, 1)
            y = random.randint(0, 1)
            
            # Compute y
            if x == 0 and y == 0:
                y = 0
            elif x == 0 and y == 1:
                y = 1
            elif x == 1 and y == 0:
                y = 1
            else:
                y = 0
            
            # If y matches the target and we need more samples for this class
            if y == target_y:
                # Create one-hot encoded output
                output_matrix = [0] * 2
                output_matrix[y] = 1
                numbers[y] += 1
                
                # Format sample as a string (e.g., "1010000")
                sample = f"{x}{y}{''.join(map(str, output_matrix))}"
                samples.append(sample)
                print(sample)
    random.shuffle(samples)

    print("Class distribution:", numbers)
    return samples

def generate_balanced_quadratic_samples(samples_per_class=1818):
    samples = []
    numbers = [0] * 11  # Track samples per class
    
    # For each target y value (0 to 10)
    for target_y in range(11):
        while numbers[target_y] < samples_per_class:
            # Random coefficients and x
            a = random.randint(0, 6)
            b = random.randint(0, 6)
            c = random.randint(0, 6)
            x = random.randint(0, 9)
            
            # Compute y
            y = a * (x ** 2) + b * x + c
            
            # If y matches the target and we need more samples for this class
            if y == target_y:
                # Create one-hot encoded output
                output_matrix = [0] * 11
                output_matrix[y] = 1
                numbers[y] += 1
                

                # Format sample as a string (e.g., "12340100000")
                sample = f"{a}{b}{c}{x}{''.join(map(str, output_matrix))}"
                samples.append(sample)
    random.shuffle(samples)

    print("Class distribution:", numbers)
    return samples

# Generate 20,000 samples (approximately 1818 per class)
with open("Data.txt", "w") as f:
    for sample in generate_balanced_quadratic_samples(300):
        f.write(sample + "\n")