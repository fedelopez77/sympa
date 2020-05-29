import torch
import sympa.math.symmetric_math as sm


def get_random_symmetric_matrices(points: int, dims: int):
    """Returns 'points' random symmetric matrices of 'dims' x 'dims'"""
    m = torch.rand(points, 2, dims, dims)
    return sm.to_symmetric(m)


def assert_equal(a: torch.tensor, b: torch.tensor):
    return torch.eq(a, b).all()


def assert_almost_equal(a, b, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def get_matrix_with_positive_eigenvalues():
    found = False
    for i in range(1000):
        m = get_random_symmetric_matrices(1, 2)
        m = sm.to_hermitian(m)
        sym_matrix = sm.to_compound_real_symmetric_from_hermitian(m)
        eigvalues, eigvectors = sm.symeig(sym_matrix)
        if torch.all(eigvalues > 0):
            print(m)
            found = True

    if not found:
        print("It couldn't find any matrix :(")
