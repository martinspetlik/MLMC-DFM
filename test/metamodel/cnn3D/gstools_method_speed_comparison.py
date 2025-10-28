import numpy as np
import gstools as gs
import time

# Define a stationary covariance model
model = gs.Gaussian(dim=3, var=1.0, len_scale=10.0)

# Grid sizes to test (you can expand this list)
grid_sizes = [16, 32, 64, 128, 160]

print("Benchmarking gstools SRF on regular 3D grids:\n")

for size in grid_sizes:
    print(f"Grid size: {size}³")

    x = y = z = np.linspace(0, 100, size)

    print("x*y*z ", x*y*z)

    # FFT mode
    start = time.time()
    srf_fft = gs.SRF(model, mode='fft', seed=42)
    _ = srf_fft.structured([x, y, z])
    fft_time = time.time() - start
    print(f"  FFT mode:     {fft_time:.4f} seconds")

    # Spectral mode with mode_no=10000
    start = time.time()
    srf_spec = gs.SRF(model, mode='spectral', mode_no=10000, seed=42)
    _ = srf_spec.structured([x, y, z])
    spec_time = time.time() - start
    print(f"  Spectral mode (mode_no=10000): {spec_time:.4f} seconds\n")