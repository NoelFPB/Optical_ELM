import numpy as np
from sklearn.datasets import load_iris
from CLEAN.iris_elm_engine import PhotonicEngine
import time
# --- 1. CONFIGURATION BASED ON YOUR 56-HEATER IMAGE ---
# Leftmost Block: Input
INPUT_ADDR = [42, 43, 44, 45] 
# Rightmost Block: 3 heaters for 3 Iris Classes
OUTPUT_WEIGHT_ADDR = [0, 1, 2] 
# All available heaters on the chip
ALL_ADDR = list(range(49))

engine = PhotonicEngine(
    input_heaters=INPUT_ADDR,
    output_heaters=OUTPUT_WEIGHT_ADDR,
    all_heaters=ALL_ADDR,
    laser_address="GPIB0::6::INSTR",
    scope_ids=['HDO1B244000779']
)

# --- 2. PREPARE IRIS DATA ---
print("Loading Iris Dataset...")
iris = load_iris()
X, y = iris.data, iris.target
# Normalize features to [0, 1] for the heaters
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# --- 3. CHARACTERIZATION (Finding the Hidden States) ---
# We run the data once with output weights at 0V to see how the 'Reservoir' reacts
H_matrix = engine.run_measurement(X_norm, gain=0.5)

# --- 4. THE ELM STEP: SOLVE FOR PHYSICAL WEIGHTS ---
print("Calculating Physical Output Weights...")
physical_beta_voltages = engine.solve_physical_weights(H_matrix, y)

# --- 5. DEPLOY TO HARDWARE ---
print(f"Deploying Weights to Heaters {OUTPUT_WEIGHT_ADDR}: {physical_beta_voltages}")
engine.bus.set(OUTPUT_WEIGHT_ADDR, physical_beta_voltages.tolist())

# --- 6. VERIFICATION (Hardware Inference) ---
# Now, if you input a sample, the light at the output ports 
# corresponds DIRECTLY to the class prediction.
test_sample = X_norm[0] # Should be Setosa (Class 0)
v_in = np.clip(2.5 + (test_sample - 0.5) * 2.0 * 0.5 * 2.4, 0.1, 4.9)

engine.laser.turn_on()
engine.bus.set(INPUT_ADDR, v_in.tolist())
time.sleep(0.5)
final_prediction = engine.scope.read_many(avg=1)
print(f"Scope Voltages (Class 0, 1, 2): {final_prediction[:3]}")
print(f"Predicted Class (Index of Max Voltage): {np.argmax(final_prediction[:3])}")
engine.laser.turn_off()