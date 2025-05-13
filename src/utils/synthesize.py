"""
synthesize neural representations for testing metrics
"""
import numpy as np
from typing import List


def generate_factor_matrix(m: int, k: int) -> np.ndarray:
    """
    generate a random factor matrix
    args:
        m (int): number of stimulus
        k (int): number of independent factors
    returns:
        a full-rank factor matrix
    """
    # sample from a random normal distribution
    rand_matrix = np.random.randn(m, k)
    # orthogonalize columns
    F, _ = np.linalg.qr(rand_matrix)

    return F


def generate_sparsity_matrix(k: int, n: int, sparsity: float) -> np.ndarray:
    """
    generate a sparse, binary matrix
    args:
        k (int): number of independent factors
        n (int): number of neurons in the representation
        sparsity (float): fraction of total matrix entries which are 0
    returns:
        a binary matrix of dimensions (k x n)
    """
    return (np.random.randn(k, n) > sparsity).astype(float)


def generate_random_normal_matrix(k: int, n: int) -> np.ndarray:
    """
    sample a matrix of given dimensions from a
    random normal distribution
    args:
        k (int): number of independent factors
        n (int): number of neurons in the representation
    returns:
        a random normal matrix of dimensions (k x n)
    """
    return np.random.randn(k, n)


def generate_neural_representations(
    k: int, m: int, n: int, num_representations: int, sparsity: float
) -> List[np.ndarray]:
    """
    synthesize neural representations by mixing independent factors
    varying in their individual sparsities
    args:
        k (int): number of independent factors
        m (int): number of stimulus
        n (int): number of neurons in the representation
        num_representations (int): number of representations to synthesize using the same
                                   sparsity matrix
        sparsity (float): fraction of total matrix entries which are 0 (higher => unit-level matching)
    returns:
        a list of neural representations synthesized using the same sparsity matrix
    """
    # generate a sparsity matrix
    sparsity_matrix = generate_sparsity_matrix(k=k, n=n, sparsity=sparsity)

    # generate a factor matrix
    factor_matrix = generate_factor_matrix(m=m, k=k)

    final_representations = []
    for _ in range(num_representations):
        x = generate_random_normal_matrix(k=k, n=n)
        # sparsify base representations
        m = sparsity_matrix * x
        # mix sparse base representations with indepent factors
        z = factor_matrix @ m
        final_representations.append(z)

    return final_representations


def concat_noisy_neurons(m: int, neuron_signal: np.ndarray, num_noisy_neurons: int):
    """
    concatenate the specified number of random neurons to a
    synthetic neural representation
    args:
        m (int):
        neuron_signal (np.ndarray):
        num_noisy_neurons (int):
    returns:
        a neural representation with the specified number of noisy neurons
        concatenated along its row dim
    """
    noise_matrix = generate_random_normal_matrix(k=m, n=num_noisy_neurons)
    neuron_signal = np.concatenate([neuron_signal, noise_matrix], axis=1)

    return neuron_signal
