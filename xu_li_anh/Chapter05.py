import numpy as np
import cv2

L = 256

# Function to create motion filter
def CreateMotionfilter(M, N):
    H = np.zeros((M, N), np.complex128)  # Updated to np.complex128
    a = 0.1
    b = 0.1
    T = 1
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi * ((u - M // 2) * a + (v - N // 2) * b)
            if np.abs(phi) < 1.0e-6:
                RE = T * np.cos(phi)
                IM = -T * np.sin(phi)
            else:
                RE = T * np.sin(phi) / phi * np.cos(phi)
                IM = -T * np.sin(phi) / phi * np.sin(phi)
            H.real[u, v] = RE
            H.imag[u, v] = IM
    return H

# Function to create motion noise
def CreateMotionNoise(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)  # Updated to np.float64
    # Step 1: DFT
    F = np.fft.fft2(f)
    # Step 2: Shift to the center of the image
    F = np.fft.fftshift(F)

    # Step 3: Create filter H
    H = CreateMotionfilter(M, N)

    # Step 4: Multiply F with H
    G = F * H

    # Step 5: Shift return
    G = np.fft.ifftshift(G)

    # Step 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L - 1)
    g = g.astype(np.uint8)
    return g

# Function to create inverse motion filter
def CreateInverseMotionfilter(M, N):
    H = np.zeros((M, N), np.complex128)  # Updated to np.complex128
    a = 0.1
    b = 0.1
    T = 1
    phi_prev = 0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi * ((u - M // 2) * a + (v - N // 2) * b)
            if np.abs(phi) < 1.0e-6:
                RE = np.cos(phi) / T
                IM = np.sin(phi) / T
            else:
                if np.abs(np.sin(phi)) < 1.0e-6:
                    phi = phi_prev
                RE = phi / (T * np.sin(phi)) * np.cos(phi)
                IM = phi / (T * np.sin(phi)) * np.sin(phi)
            H.real[u, v] = RE
            H.imag[u, v] = IM
            phi_prev = phi
    return H

# Function to denoise motion
def DenoiseMotion(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)  # Updated to np.float64
    # Step 1: DFT
    F = np.fft.fft2(f)
    # Step 2: Shift to the center of the image
    F = np.fft.fftshift(F)

    # Step 3: Create filter H
    H = CreateInverseMotionfilter(M, N)

    # Step 4: Multiply F with H
    G = F * H

    # Step 5: Shift return
    G = np.fft.ifftshift(G)

    # Step 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L - 1)
    g = g.astype(np.uint8)
    return g
