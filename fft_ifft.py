import numpy as np


def DFT_1D(input_matrix):
    input_matrix = np.asarray(input_matrix, dtype=complex)
    N = input_matrix.shape[0]
    X_copy = input_matrix.copy()
    i = 0
    while i<N:
        u = i
        sum = 0
        for j in range(N):
            x = j
            tmp = input_matrix[x]*np.exp(-2j*np.pi*x*u*np.divide(1, N, dtype=complex))
            sum += tmp
        X_copy[u] = sum
        i = i+1
    return X_copy

def inverseDFT_1D(input_matrix):
    input_matrix = np.asarray(input_matrix, dtype=complex)
    N = input_matrix.shape[0]
    input_matrix = np.zeros(N, dtype=complex)
    i = 0
    while i<N:
        x = i
        sum = 0
        for j in range(N):
            u = j
            tmp = input_matrix[u]*np.exp(2j*np.pi*x*u*np.divide(1, N, dtype=complex))
            sum += tmp
        input_matrix[x] = np.divide(sum, N, dtype=complex)
        i = i+1

    return input_matrix


def FFT_1D(input_matrix):
    """ use recursive method to speed up"""
    input_matrix = np.asarray(input_matrix, dtype=complex)
    N = input_matrix.shape[0]
    minDivideSize = 4

    if N % 2 != 0:
        raise ValueError("the input size must be 2^n")

    if N <= minDivideSize:
        return DFT_1D(input_matrix)
    else:
        X_even = FFT_1D(input_matrix[::2])  # compute the even part
        X_odd = FFT_1D(input_matrix[1::2])  # compute the odd part
        W_ux_2k = np.exp(-2j * np.pi * np.arange(N) / N)

        f_u = X_even + X_odd * W_ux_2k[:N//2]

        f_u_plus_k = X_even + X_odd * W_ux_2k[N//2:]

        fft_mat = np.concatenate([f_u, f_u_plus_k])

    return fft_mat


def inverseFFT_1D(Y):
    """ use recursive method to speed up"""
    Y = np.asarray(Y, dtype=complex)
    fu_conjugate = np.conjugate(Y)

    input_matrix = FFT_1D(fu_conjugate)

    input_matrix = np.conjugate(input_matrix)
    input_matrix = input_matrix / Y.shape[0]

    return input_matrix


def FFT_2D(input_matrix):
    h, w = input_matrix.shape[0], input_matrix.shape[1]

    Y = np.zeros(input_matrix.shape, dtype=complex)

    if len(input_matrix.shape) == 2:
        for i in range(h):
            Y[i, :] = FFT_1D(input_matrix[i, :])

        for i in range(w):
            Y[:, i] = FFT_1D(Y[:, i])
    elif len(input_matrix.shape) == 3:
        for ch in range(3):
            Y[:, :, ch] = FFT_2D(input_matrix[:, :, ch])

    return Y


def inverseDFT_2D(Y):
    h, w = Y.shape[0], Y.shape[1]

    input_matrix = np.zeros(Y.shape, dtype=complex)

    if len(Y.shape) == 2:
        for i in range(h):
            input_matrix[i, :] = inverseDFT_1D(Y[i, :])

        for i in range(w):
            input_matrix[:, i] = inverseDFT_1D(input_matrix[:, i])

    elif len(Y.shape) == 3:
        for ch in range(3):
            input_matrix[:, :, ch] = inverseDFT_2D(Y[:, :, ch])

    input_matrix = np.real(input_matrix)
    return input_matrix


def inverseFFT_2D(Y):
    h, w = Y.shape[0], Y.shape[1]

    input_matrix = np.zeros(Y.shape, dtype=complex)

    if len(Y.shape) == 2:
        for i in range(h):
            input_matrix[i, :] = inverseFFT_1D(Y[i, :])

        for i in range(w):
            input_matrix[:, i] = inverseFFT_1D(input_matrix[:, i])

    elif len(Y.shape) == 3:
        for ch in range(3):
            input_matrix[:, :, ch] = inverseFFT_2D(Y[:, :, ch])

    input_matrix = np.real(input_matrix)
    return input_matrix

