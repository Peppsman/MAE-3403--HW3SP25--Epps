# MAE 3403 HW3BSP25 Eppa, Patrick (PBE)


import math  # Import the math module


def compute_probability(m, z):
    """
    Compute the right-hand side of the given equation using the t-distribution without scipy.

    Parameters:
    m (int): Degrees of freedom
    z (float): Value of z

    Returns:
    float: Probability
    """
    k_m = math.gamma(0.5 * m + 0.5) / (math.sqrt(m * math.pi) * math.gamma(0.5 * m))
    # Calculate the normalization constant K_m

    t_cdf = 0  # Initialize the cumulative distribution function (CDF) for the t-distribution

    # Use numerical integration (trapezoidal rule) to calculate the CDF
    n = 1000  # Number of integration points
    dz = z / n  # Step size
    for i in range(n):
        x = i * dz
        t_cdf += (1 + (x ** 2 / m)) ** (-(m + 1) / 2) * dz
    t_cdf = 0.5 + t_cdf / math.sqrt(math.pi * m)

    result = k_m * t_cdf
    # Multiply by the normalization constant K_m

    return result  # Return the probability


def main():
    """
    Main function to prompt user input and compute probabilities.

    The function prompts the user to input degrees of freedom and z values,
    then computes and prints the corresponding probabilities.
    """
    degrees_of_freedom = [7, 11, 15]
    # List of degrees of freedom to test

    for m in degrees_of_freedom:
        print(f"Degrees of freedom (m): {m}")
        # Print the current degrees of freedom

        z_values = input(f"Enter three z values for m={m}, separated by spaces: ").split()
        # Prompt the user to input three z values separated by spaces, then split the input into a list

        z_values = [float(z) for z in z_values]
        # Convert the z values to floats

        for z in z_values:
            probability = compute_probability(m, z)
            # Compute the probability for the given m and z

            print(f"z = {z}, Probability = {probability:.4f}")
            # Print the z value and corresponding probability with 4 decimal places


if __name__ == "__main__":
    main()
    # Run the main function when the script is executed
