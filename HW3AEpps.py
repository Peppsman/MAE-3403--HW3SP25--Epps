#MAE 3403 HW3ASP25 Epps, Patrick (PBE)


import numpy as np

def simpsons_rule(f, a, b, n):
    """
    Perform numerical integration using Simpson's 1/3 method.

    Parameters:
    f (function): The function to integrate.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals (must be even).

    Returns:
    float: The integral of f from a to b.
    """
    if n % 2 == 1:
        n += 1  # Ensure n is even

    h = (b - a) / n
    integral = f(a) + f(b)

    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    for i in range(2, n - 1, 2):
        integral += 2 * f(a + i * h)

    integral *= h / 3
    return integral

def normal_pdf(x, mu, sigma):
    """
    Normal probability density function.

    Parameters:
    x (float): The variable.
    mu (float): The mean.
    sigma (float): The standard deviation.

    Returns:
    float: The value of the normal PDF at x.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def compute_probability(mu, sigma, c, double_sided=True):
    """
    Compute the probability for the given parameters.

    Parameters:
    mu (float): The mean.
    sigma (float): The standard deviation.
    c (float): The value of c.
    double_sided (bool): Whether to compute double-sided probability.

    Returns:
    float: The computed probability.
    """
    if double_sided:
        a = mu - (c - mu)
        b = mu + (c - mu)
    else:
        a = mu - 5 * sigma
        b = c

    return simpsons_rule(lambda x: normal_pdf(x, mu, sigma), a, b, 1000)

def secant_method(f, a, b, tol=1e-6, max_iter=100):
    """
    Use the Secant method to find the root of the function f.

    Parameters:
    f (function): The function whose root we want to find.
    a (float): The initial guess.
    b (float): The second guess.
    tol (float): The tolerance for convergence.
    max_iter (int): The maximum number of iterations.

    Returns:
    float: The value of the root.
    """
    for _ in range(max_iter):
        fa = f(a)
        fb = f(b)
        if abs(fb - fa) < tol:
            return b
        c = b - fb * (b - a) / (fb - fa)
        if abs(c - b) < tol:
            return c
        a, b = b, c
    return b

def main():
    # Solicit input from the user
    mu = float(input("Enter the mean (μ): "))
    sigma = float(input("Enter the standard deviation (σ): "))
    mode = input("Are you specifying 'c' and seeking 'P' or specifying 'P' and seeking 'c'? (Enter 'c' or 'P'): ").strip().lower()

    if mode == 'c':
        c = float(input("Enter the value of c: "))
        double_sided = input("Is it double-sided probability? (yes or no): ").strip().lower() == 'yes'
        P = compute_probability(mu, sigma, c, double_sided)
        print(f"The computed probability is: {P:.6f}")

    elif mode == 'p':
        P = float(input("Enter the desired probability: "))
        double_sided = input("Is it double-sided probability? (yes or no): ").strip().lower() == 'yes'
        target_function = lambda c: compute_probability(mu, sigma, c, double_sided) - P
        c = secant_method(target_function, mu, mu + sigma)
        print(f"The value of c that matches the desired probability is: {c:.6f}")

    else:
        print("Invalid input. Please enter 'c' or 'P'.")

if __name__ == "__main__":
    main()
    # Run the main function when the script is executed
