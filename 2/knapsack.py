import random
import mlrose_hiive as ml
from util import set_seed, start_time, plot_runner


def ceil(x):
    return int(-(-x // 1))


def knapsack(input, max_pct=0.5, weight=None, value=None):
    max_weight = input if weight is None else weight
    max_value = input if value is None else value
    weights = random.choices(list(range(1, max_weight+1)), k=input)
    values = random.choices(list(range(1, max_value+1)), k=input)
    fitness = ml.Knapsack(weights, values, max_pct)
    problem = ml.DiscreteOpt(
        length=input,
        fitness_fn=fitness,
        maximize=True,
        max_val=2,
    )
    return problem


def rhc(problem):
    print("Running knapsack RHC...")
    end_time = start_time()
    results = ml.random_hill_climb(
        problem,
        max_attempts=100,
        restarts=10,
        curve=True,
    )
    time = end_time()
    _, fitness, curve = results
    _, iterations = zip(*curve)
    return time, fitness, iterations[-1]


def sa(problem):
    print("Running knapsack SA...")
    end_time = start_time()
    results = ml.simulated_annealing(
        problem,
        schedule=ml.ExpDecay(),
        max_attempts=100,
        max_iters=1000,
        curve=True,
    )
    time = end_time()
    _, fitness, curve = results
    _, iterations = zip(*curve)
    return time, fitness, iterations[-1]


def ga(problem):
    print("Running knapsack GA...")
    end_time = start_time()
    results = ml.genetic_alg(
        problem,
        pop_size=200,
        mutation_prob=0.2,
        max_attempts=100,
        max_iters=1000,
        curve=True,
    )
    time = end_time()
    _, fitness, curve = results
    _, iterations = zip(*curve)
    return time, fitness, iterations[-1]


def mimic(problem):
    print("Running knapsack MIMIC...")
    end_time = start_time()
    results = ml.mimic(
        problem,
        pop_size=200,
        keep_pct=0.2,
        max_attempts=10,
        max_iters=1000,
        curve=True,
    )
    time = end_time()
    _, fitness, curve = results
    _, iterations = zip(*curve)
    return time, fitness, iterations[-1]


def main():
    plot_runner(
        "Knapsack",
        knapsack,
        {
            "RHC": rhc,
            "SA": sa,
            "GA": ga,
            "MIMIC": mimic,
        },
        # [*range(2, 10, 2)],
    )


if __name__ == "__main__":
    set_seed()
    main()
