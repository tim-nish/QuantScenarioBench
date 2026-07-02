class QuantScenarioBenchSolverError(Exception):
    """Raised when the Optimizer Solver Layer fails to converge (AD-14).

    Signals a solve_allocation(...) failure — the caller must not receive
    a degenerate or unconverged weight vector in this case.
    """
    pass
