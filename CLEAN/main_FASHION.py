import numpy as np
from sklearn.datasets import fetch_openml
from CLEAN.engine import PhotonicEngine

# --- 1. CONFIGURATION ---
# --- CONFIGURATION: The Sandwich ---
# 7 Mesh | 14 Input | 7 Mesh | 14 Input | 7 Mesh (Total 49)
group1 = list(range(7, 21))   # First 14
group2 = list(range(28, 42))  # Second 14

INPUT_HEATERS = group1 + group2  # Total 28 addresses
ROW_BANDS = 7               
ALL_HEATERS   = list(range(49))

# The engine will automatically pick up m1, m2, and m3 as the mesh!
WAVELENGTHS   = [1544.0, 1548.0, 1552.0, 1556.0]  
SAMPLES_PER_CLASS = 20 # Keep it small for quick testing
GAIN = 0.4
# --- 2. SETUP & DATA ---
engine = PhotonicEngine(
    input_heaters=INPUT_HEATERS, 
    all_heaters=ALL_HEATERS,
    laser_address="GPIB0::6::INSTR",
    scope_ids=['HDO1B244000779']
)

print("Loading Fashion-MNIST...")
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X_raw, y_raw = data.data / 255.0, data.target.astype(int)

# Filter for a small balanced subset
indices = np.hstack([np.where(y_raw == i)[0][:SAMPLES_PER_CLASS] for i in range(10)])
X_sub, y_sub = X_raw[indices], y_raw[indices]

# Pre-encode images spatially
X_encoded = [engine.encode_image(img, row_bands=ROW_BANDS) for img in X_sub]
# --- 3. RUN EXPERIMENT ---
print(f"Starting Measurement: {len(X_sub)} samples x {len(WAVELENGTHS)} wavelengths")
X_features = engine.run_measurement(X_encoded, gain=GAIN, wavelengths=WAVELENGTHS)

# Training on concatenated wavelengths (The Multi-λ ELM boost)
X_concat = X_features.reshape(len(y_sub), -1) 
model, acc, test_data = engine.train_elm(X_concat, y_sub)

# --- 4. RESULTS ---
print(f"Final Accuracy: {acc*100:.2f}%")
engine.save_and_plot(model, test_data, acc, GAIN, WAVELENGTHS, SAMPLES_PER_CLASS, {"L": len(WAVELENGTHS)}, name="fashion_multiplex")

engine.save_results(X_features, y_sub, GAIN, WAVELENGTHS, name="intercalated_dual")