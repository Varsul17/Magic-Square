import numpy as np
import random
import time

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon_start = 1.0  # Initial epsilon value for exploration
epsilon_decay = 0.99  # Decay rate for epsilon
epsilon_min = 0.01  # Minimum epsilon value
episodes = 20000  # Number of training episodes


def magic_sum(n):
    """
    Calculate the magic sum for an N×N magic square.
    Input: n (size of the magic square)
    Output: The magic constant (integer)
    """
    return (n * (n ** 2 + 1)) // 2


def get_reward(grid, n):
    """
    Calculate the reward based on the current state of the grid.
    Input: grid (2D list representing the magic square), n (size of the grid)
    Output: reward (integer score based on closeness to a valid magic square)
    """
    target = magic_sum(n)  # Target sum for rows, columns, and diagonals

    # Compute sums of completed rows, columns, and diagonals
    row_sums = [sum(row) for row in grid if 0 not in row]
    col_sums = [sum(col) for col in zip(*grid) if 0 not in col]
    diag_sums = [sum(grid[i][i] for i in range(n)), sum(grid[i][n - 1 - i] for i in range(n))]

    reward = 0
    # Reward based on closeness to target sum
    for s in row_sums + col_sums + diag_sums:
        reward += max(0, 10 - abs(target - s))

    # Extra reward if the entire grid forms a valid magic square
    if len(row_sums) == n and len(col_sums) == n and all(s == target for s in row_sums + col_sums + diag_sums):
        reward += 100

    return reward


def solve_magic_square(n):
    """
    Train an agent using Q-learning to solve an N×N magic square.
    Input: n (size of the magic square)
    Output: Prints the learned magic square and training statistics.
    """
    numbers = list(range(1, n ** 2 + 1))  # Possible numbers to use in the magic square
    Q_table = {}  # Initialize Q-table
    epsilon = epsilon_start  # Initialize epsilon for exploration-exploitation tradeoff

    train_start_time = time.time()  # Start training timer

    # Training loop
    for episode in range(episodes):
        grid = [[0] * n for _ in range(n)]  # Reset grid for each episode
        available_numbers = numbers.copy()  # Reset available numbers

        for _ in range(n ** 2):
            state = tuple(tuple(row) for row in grid)  # Convert grid to a hashable state

            # Choose action (number) using epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                num = random.choice(available_numbers)  # Explore
            else:
                num = max(available_numbers, key=lambda x: Q_table.get((state, x), 0))  # Exploit

            # Get list of empty cells
            empty_cells = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]
            if not empty_cells:
                break
            r, c = random.choice(empty_cells)  # Pick a random empty cell

            grid[r][c] = num  # Place number in grid
            available_numbers.remove(num)  # Remove number from available choices

            reward = get_reward(grid, n)  # Get reward for the new state

            new_state = tuple(tuple(row) for row in grid)  # Update state

            # Update Q-value using Bellman equation
            best_next_q = max([Q_table.get((new_state, a), 0) for a in available_numbers], default=0)
            Q_table[(state, num)] = Q_table.get((state, num), 0) + alpha * (
                        reward + gamma * best_next_q - Q_table.get((state, num), 0))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon

    train_end_time = time.time()  # End training timer
    train_time = train_end_time - train_start_time  # Compute training duration

    # Testing phase (use learned policy to construct the magic square)
    test_start_time = time.time()  # Start inference timer

    test_grid = [[0] * n for _ in range(n)]  # Reset grid
    available_numbers = numbers.copy()  # Reset available numbers

    for _ in range(n ** 2):
        state = tuple(tuple(row) for row in test_grid)  # Convert grid to state
        num = max(available_numbers, key=lambda x: Q_table.get((state, x), 0))  # Choose best action
        empty_cells = [(r, c) for r in range(n) for c in range(n) if test_grid[r][c] == 0]
        r, c = random.choice(empty_cells)  # Pick random empty cell
        test_grid[r][c] = num  # Place number in grid
        available_numbers.remove(num)  # Remove number from available choices

    test_end_time = time.time()  # End inference timer
    test_time = test_end_time - test_start_time  # Compute inference duration

    # Print results
    print(f"\n🔹 Learned {n}×{n} Magic Square:")
    for row in test_grid:
        print(row)

    print(f"\n⏳ Training Time: {train_time:.2f} seconds")
    print(f"⏳ Inference Time: {test_time:.6f} seconds")


def main():
    # Ask the user for the size of the magic square
    while True:
        try:
            N = int(input("Enter the size of the magic square (N×N): "))  # Get user input
            if N < 3:
                print("Magic squares are only valid for N ≥ 3. Try again.")  # Validate input
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    solve_magic_square(N)  # Solve the magic square


if __name__ == "__main__":
    main()
