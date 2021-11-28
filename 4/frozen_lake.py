from util import set_seed
from solvers import q as q_solver, policy as policy_solver, value as value_solver

def run():
    pass

def policy():
    run(policy_solver)

def value():
    run(value_solver)

def q():
    run(q_solver)

def main():
    policy()
    value()
    q()


if __name__ == "__main__":
    set_seed()
    main()
