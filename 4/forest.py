from util import set_seed, plt, plot
from hiive.mdptoolbox.example import forest
from solvers import q as q_solver, policy as policy_solver, value as value_solver


def run(name, solver):
    sizes = [*range(2, 10, 2), *range(10, 50, 10), *range(50, 1001, 100)]
    times = []
    iterations = []
    solutions = []
    times = []
    mean_values = []
    max_values = []
    errors = []
    for size in sizes:
        P, R = forest(size, 3, 2, 0.1)
        _, i, p, t, v_mean, v_max, e = solver(P, R)
        iterations.append(i)
        solutions.append(p.count(1))
        times.append(t)
        mean_values.append(v_mean)
        max_values.append(v_max)
        errors.append(e)
    return sizes, times, iterations, times, solutions, times, mean_values, max_values, errors


def policy():
    return run("Forest Policy Iteration", policy_solver)


def value():
    return run("Forest Value Iteration", value_solver)


def q():
    return run("Forest Q Learner", q_solver)


def runner():
    policy_sizes, policy_times, policy_iterations, policy_times, policy_solutions, policy_times, policy_mean_values, policy_max_values, policy_errors = policy()
    value_sizes, value_times, value_iterations, value_times, value_solutions, value_times, value_mean_values, value_max_values, value_errors = value()
    q_sizes, q_times, q_iterations, q_times, q_solutions, q_times, q_mean_values, q_max_values, q_errors = q()

    plt(policy_sizes, policy_iterations, label="Policy Iterations")
    plt(value_sizes, value_iterations, label="Value Iterations")
    plt(q_sizes, q_iterations, label="Q Learner Iterations")
    plot(
        'Forest Value, Policy & Q Learner Iterations per Size',
        xlabel="Size of Fire",
        ylabel="Iteration",
    )

    plt(policy_sizes, policy_solutions, label="Policy Solutions")
    plt(value_sizes, value_solutions, label="Value Solutions")
    plt(q_sizes, q_solutions, label="Q Learner Solutions")
    plot(
        'Forest Value, Policy & Q Learner Solutions per Size',
        xlabel="Size of Fire",
        ylabel="Solution",
    )

    plt(policy_sizes, policy_times, label="Policy Times")
    plt(value_sizes, value_times, label="Value Times")
    plt(q_sizes, q_times, label="Q Learner Times")
    plot(
        'Forest Value, Policy & Q Learner Times per Size',
        xlabel="Size of Fire",
        ylabel="Time (s)",
    )

    plt(policy_sizes, policy_mean_values, label="Policy Mean Values")
    plt(value_sizes, value_mean_values, label="Value Mean Values")
    plt(q_sizes, q_mean_values, label="Q Learner Mean Values")
    plt(policy_sizes, policy_max_values, label="Policy Max Values")
    plt(value_sizes, value_max_values, label="Value Max Values")
    plt(q_sizes, q_max_values, label="Q Learner Max Values")
    plot(
        'Forest Value, Policy & Q Learner Values per Size',
        xlabel="Size of Fire",
        ylabel="Value",
    )

    plt(policy_sizes, policy_errors, label="Policy Errors")
    plt(value_sizes, value_errors, label="Value Errors")
    plt(q_sizes, q_errors, label="Q Learner Errors")
    plot(
        'Forest Value, Policy & Q Learner Errors per Size',
        xlabel="Size of Fire",
        ylabel="Error",
    )


def main():
    runner()


if __name__ == "__main__":
    set_seed()
    main()
