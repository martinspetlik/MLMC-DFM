import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


def empirical_variogram(pos, values, nbins=30):

    D = squareform(pdist(pos))
    print("D ", D)
    V = squareform(pdist(values[:, None], metric='sqeuclidean')) / 2
    bins = np.linspace(0, D.max(), nbins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    variogram = []

    print("bin centers ", bin_centers)

    for i in range(nbins):
        mask = (D >= bins[i]) & (D < bins[i + 1])
        if np.any(mask):
            variogram.append(np.mean(V[mask]))
        else:
            variogram.append(np.nan)
    return bin_centers, np.array(variogram)


def variogram_model(h, C, L):
    return C * (1 - np.exp(-h / L))  # Exponential model


def fit_variogram_model(distances, gamma_vals):
    mask = ~np.isnan(gamma_vals)
    popt, _ = curve_fit(variogram_model, distances[mask], gamma_vals[mask], bounds=(0, np.inf))
    return popt  # C (sill), L (correlation length)


def get_corr_lengths(coords, cond_tns, use_log=False):
    from bgem.upscale import fem_plot, fem, voigt_to_tn, tn_to_voigt

    components = tn_to_voigt(cond_tns)
    print("components.shape ", components.shape)

    if use_log:
        components[..., 0] = np.log(components[..., 0])
        components[..., 1] = np.log(components[..., 1])
        components[..., 2] = np.log(components[..., 2])

    results = {}
    for i in range(components.shape[-1]):
        values = np.squeeze(components[..., i])
        print("values.shape ", values.shape)
        print("coords.shape ", coords.shape)
        dists, gamma = empirical_variogram(coords, values)
        C, L = fit_variogram_model(dists, gamma)

        print("Component {}, still: {}, correlation_length: {}".format(i, C, L))
        results[i] = {'sill': C, 'correlation_length': L}

        # Optional plot
        plt.figure()
        plt.plot(dists, gamma, 'o', label='Empirical')
        plt.plot(dists, variogram_model(dists, C, L), '-', label=f'Model: L={L:.2f}')
        plt.xlabel("Distance")
        plt.ylabel("Semivariance")
        plt.title(f"Variogram of $T_{{{i}}}$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


