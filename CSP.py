import time
import psutil
import os
from ortools.sat.python import cp_model


def solve_magic_square(n):
    model = cp_model.CpModel()

    # Define variables: An n x n grid with values from 1 to n^2
    cells = [[model.NewIntVar(1, n ** 2, f'cell_{i}_{j}') for j in range(n)] for i in range(n)]

    # Constraint: All numbers must be distinct
    model.AddAllDifferent([cells[i][j] for i in range(n) for j in range(n)])

    # Magic sum formula
    magic_sum = n * (n ** 2 + 1) // 2

    # Row constraints
    for i in range(n):
        model.Add(sum(cells[i][j] for j in range(n)) == magic_sum)

    # Column constraints
    for j in range(n):
        model.Add(sum(cells[i][j] for i in range(n)) == magic_sum)

    # Main diagonal constraint
    model.Add(sum(cells[i][i] for i in range(n)) == magic_sum)

    # Secondary diagonal constraint
    model.Add(sum(cells[i][n - i - 1] for i in range(n)) == magic_sum)

    # Measure memory and time before solving
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    start_time = time.time()

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Measure memory and time after solving
    mem_after = process.memory_info().rss
    end_time = time.time()

    # Compute memory and time usage
    mem_used = (mem_after - mem_before) / (1024 * 1024)  # Convert to MB
    time_taken = end_time - start_time  # Time in seconds

    # Print the solution if found
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        print(f"\nMagic Square of size {n}x{n}:\n")
        for i in range(n):
            print([solver.Value(cells[i][j]) for j in range(n)])
        print(f"\nMemory used: {mem_used:.4f} MB")
        print(f"Time taken: {time_taken:.4f} seconds\n")
    else:
        print("No solution found!")


# Ask for user input
n = int(input("Enter the size of the magic square (n): "))
solve_magic_square(n)
