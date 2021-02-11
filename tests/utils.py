import torch
import sympa.math.csym_math as sm


def get_random_symmetric_matrices(points: int, dims: int):
    """Returns 'points' random symmetric matrices of 'dims' x 'dims'"""
    m = torch.rand(points, 2, dims, dims)
    return sm.to_symmetric(m)


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
