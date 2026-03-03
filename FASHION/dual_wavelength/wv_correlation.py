import numpy as np
import os
import itertools
import matplotlib.pyplot as plt

def analyze_multi_correlation(filename, plot=True):
    """
    Loads an NPZ file and analyzes correlation between ALL wavelengths present.
    """
    # 1. Load the file
    if not os.path.exists(filename):
        print(f"[ERROR] File not found: {filename}")
        return

    try:
        data = np.load(filename)
        X_stack = data["X_stack"]      # Shape: (N, L, D) -> (Samples, Wavelengths, Features)
        wavelengths = data["wavelengths"]
        
        print(f"\n[LOADED] {filename}")
        print(f"Samples: {X_stack.shape[0]} | Wavelengths: {len(wavelengths)} | Features: {X_stack.shape[2]}")
        print(f"Wavelength Values: {wavelengths} nm")

        if X_stack.shape[1] < 2:
            print("[ERROR] Need at least 2 wavelengths to calculate correlation.")
            return

    except Exception as e:
        print(f"[ERROR] Could not load file. \n{e}")
        return

    n_lambdas = len(wavelengths)
    n_features = X_stack.shape[2]
    
    # 2. Compute Pairwise Correlations (Global Average)
    # We want to know: "How similar is Lambda A to Lambda B?"
    # We calculate the correlation of the ENTIRE feature set (flattened) for a robust metric.
    
    correlation_matrix = np.zeros((n_lambdas, n_lambdas))
    
    print(f"\n{'PAIR':<20} | {'CORRELATION':<15} | {'VERDICT'}")
    print("-" * 60)
    
    # Generate all pairs (0,1), (0,2), (1,2), etc.
    pairs = list(itertools.combinations(range(n_lambdas), 2))
    
    for i, j in pairs:
        # Flatten all samples and features into one giant vector for this wavelength
        # This compares the "Global Information Content" of the two wavelengths
        vec_i = X_stack[:, i, :].flatten()
        vec_j = X_stack[:, j, :].flatten()
        
        corr = np.corrcoef(vec_i, vec_j)[0, 1]
        correlation_matrix[i, j] = corr
        correlation_matrix[j, i] = corr # Symmetric
        
        # Diagnosis
        if corr > 0.90: verdict = "BAD (Redundant)"
        elif corr < 0.60: verdict = "EXCELLENT (Diverse)"
        else: verdict = "OK (Partial Overlap)"
        
        print(f"{wavelengths[i]} vs {wavelengths[j]:<6} | {corr:.4f}          | {verdict}")

    # Fill diagonal with 1.0
    np.fill_diagonal(correlation_matrix, 1.0)
    
    avg_diversity = np.mean([correlation_matrix[i,j] for i,j in pairs])
    print("-" * 60)
    print(f"Global Average Inter-Lambda Correlation: {avg_diversity:.4f}")

    # 3. Plot Heatmap (Perfect for Poster)
    if plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label="Pearson Correlation")
        
        # Axis labels
        ticks = np.arange(n_lambdas)
        plt.xticks(ticks, wavelengths, rotation=45)
        plt.yticks(ticks, wavelengths)
        plt.title("Wavelength Feature Correlation")
        
        # Text annotations on grid
        for i in range(n_lambdas):
            for j in range(n_lambdas):
                text = f"{correlation_matrix[i, j]:.2f}"
                plt.text(j, i, text, ha='center', va='center', 
                         color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        plt.show()
        print("[INFO] Correlation matrix plotted.")

if __name__ == "__main__":
    # Prompt user for file
    #print("Paste the path to your .npz file (e.g., FASHION/dual_wavelength/multi_lambda_....npz):")
    fname = 'multi_lambda_20251118_205553.npz'
    
    analyze_multi_correlation(fname)