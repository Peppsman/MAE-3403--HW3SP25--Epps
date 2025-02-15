#MAE 3403 HW3ASP25 Epps, Patrick (PBE)

# I used copilet to create the scafolding and added values as needed

import math
import random


def normal_pdf(x, mu, sigma):
    """
    Compute the normal probability density function (PDF).
    """
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def simpsons_rule(f, a, b, n, mu, sigma):
    """
    Perform numerical integration using Simpson's 1/3 method.
    """
    if n % 2 == 1:
        n += 1  # n must be even
    h = (b - a) / n
    integral = f(a, mu, sigma) + f(b, mu, sigma)

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            integral += 2 * f(x, mu, sigma)
        else:
            integral += 4 * f(x, mu, sigma)

    integral *= h / 3
    return integral


def normal_probability(mu, sigma, c, lower_bound=True):
    """
    Compute the probability for the normal distribution.
    """
    if lower_bound:
        a, b = mu - 5 * sigma, c
    else:
        a, b = mu - (c - mu), mu + (c - mu)
    return simpsons_rule(normal_pdf, a, b, 1000, mu, sigma)


def secant_method(f, target, x0, x1, tol=1e-6, max_iter=100):
    """
    Use the Secant method to find the value of x that matches the target value.
    """
    for _ in range(max_iter):
        fx0 = f(x0) - target
        fx1 = f(x1) - target
        if abs(fx1 - fx0) < tol:
            break
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x2
        if abs(f(x1) - target) < tol:
            return x1
    return x1


def main():
    """
    Main function to interact with the user, obtain inputs, and compute probabilities or values of c.
    """
    mu = float(input("Enter the mean (μ): "))
    sigma = float(input("Enter the standard deviation (σ): "))
    choice = input("Are you specifying c and seeking P or specifying P and seeking c? (c/P): ").strip().lower()

    if choice == 'c':
        c = float(input("Enter the value of c: "))
        single_sided = input("Do you want a single-sided probability? (yes/no): ").strip().lower() == 'yes'
        prob = normal_probability(mu, sigma, c, lower_bound=single_sided)
        print(f"Probability: {prob}")
    elif choice == 'p':
        target_prob = float(input("Enter the desired probability: "))
        single_sided = input("Do you want a single-sided probability? (yes/no): ").strip().lower() == 'yes'
        lower_bound = single_sided

        def prob_diff(c):
            return normal_probability(mu, sigma, c, lower_bound=lower_bound)

        c = secant_method(prob_diff, target_prob, mu - 5 * sigma, mu + 5 * sigma)
        print(f"Value of c: {c}")
    else:
        print("Invalid choice. Please specify 'c' or 'P'.")


if __name__ == "__main__":
    main()

