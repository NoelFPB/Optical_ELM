import os, time, numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

# Hardware Libs (Assumed path)
from Lib.scope import Rigol_Scopes
from Lib.DualBoard import DualAD5380Controller
from Lib.laser import LaserSource

class PhotonicEngine:
    def __init__(self, input_heaters, output_heaters, all_heaters, laser_address, scope_ids):
        self.inputs = list(input_heaters)
        self.outputs = list(output_heaters)
        # The "Hidden" mesh is everything not used for input or physical output weights
        self.mesh = [h for h in all_heaters if h not in self.inputs and h not in self.outputs]
        
        # Hardware Init
        self.scope = Rigol_Scopes([1,2,3,4], [1,2,3], serial_scope1=scope_ids[0])
        self.bus = DualAD5380Controller()
        self.laser = LaserSource(laser_address)
        
        # 1. Initialize Reservoir (Fixed Random States)
        rng = np.random.default_rng(42)
        mesh_vs = [float(rng.uniform(0.5, 4.5)) for _ in self.mesh]
        self.bus.set(self.mesh, mesh_vs)
        
        # 2. Set Output Weights to a neutral state (e.g., 0V) for initial characterization
        self.bus.set(self.outputs, [0.0] * len(self.outputs))

    def run_measurement(self, X_data, wavelength=1550.0, v_bias=2.5, gain=0.4):
        """
        Passes data through the chip and records the optical response.
        """
        X_features = []
        self.laser.turn_on(settle=2)
        self.laser.set_wavelength(wavelength, settle=1)

        print(f"Measuring {len(X_data)} samples...")
        for i, row in enumerate(X_data):
            # Scale input features to heater voltages (0.1V to 4.9V)
            v_in = np.clip(v_bias + (np.asarray(row) - 0.5) * 2.0 * gain * 2.4, 0.1, 4.9)
            self.bus.set(self.inputs, v_in.tolist())
            
            # Record the 4 channels from the scope
            # These are the "Hidden States" H
            X_features.append(self.scope.read_many(avg=1))
            
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(X_data)}")

        self.laser.turn_off()
        return np.array(X_features)

    def solve_physical_weights(self, H, y):
        """
        Calculates the Beta weights for the physical output layer.
        """
        # One-hot encode the targets (Setosa=[1,0,0], etc.)
        num_classes = len(np.unique(y))
        T = np.zeros((len(y), num_classes))
        for i, val in enumerate(y):
            T[i, val] = 1

        # Ridge Solve: Beta = (H^T H + lambda*I)^-1 H^T T
        # We use a simple pinv here which is the ELM standard
        beta = np.linalg.pinv(H) @ T 
        
        # Scale Beta to valid Hardware Voltages (0.1V - 4.9V)
        # We take the mean weight for each class to represent that output heater
        physical_beta = np.mean(beta, axis=0) 
        beta_v = np.interp(physical_beta, (physical_beta.min(), physical_beta.max()), (0.5, 4.5))
        
        return beta_v