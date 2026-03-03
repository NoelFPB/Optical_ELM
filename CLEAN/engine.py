import os, time, numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Import your custom hardware libs
from Lib.scope import Rigol_Scopes
from Lib.DualBoard import DualAD5380Controller
from Lib.laser import LaserSource

class PhotonicEngine:
    def __init__(self, input_heaters, all_heaters, laser_address, scope_ids):
        self.inputs = list(input_heaters)
        self.mesh = [h for h in all_heaters if h not in self.inputs]
        
        # Hardware Init
        self.scope = Rigol_Scopes([1,2,3,4], [1,2,3], serial_scope1=scope_ids[0])
        self.bus = DualAD5380Controller()
        self.laser = LaserSource(laser_address)
        
        # Set Random Mesh State (The "Reservoir")
        rng = np.random.default_rng(42)
        mesh_vs = [float(rng.uniform(0.5, 4.5)) for _ in self.mesh]
        self.bus.set(self.mesh, mesh_vs)


    def encode_image(self, img_flat, row_bands=7):
        # 1. Reshape to original 28x28
        img = img_flat.reshape(28, 28)
        
        # 2. Split 28 rows into 7 updates (Each update = 4 rows)
        bands = np.array_split(img, row_bands, axis=0)
        
        encoded_bands = []
        for b in bands:
            # b has shape (4, 28). We want to turn this into 28 unique values.
            # 3. Compress 28 columns down to 7
            # Result: (4 rows, 7 columns) = 28 unique spatial features
            update_features = b.reshape(4, 7, 4).mean(axis=2).flatten()
            
            # No np.concatenate needed anymore! 
            # Every heater from 1 to 28 gets a unique piece of the image.
            encoded_bands.append(update_features)
            
        return encoded_bands # 7 updates x 28 unique values
            

    def run_measurement(self, X_images, wavelengths, v_bias=2.5, gain=0.4):
        X_stack = []
        self.laser.turn_on(settle=2)
        
        total_wavelengths = len(wavelengths)
        total_images = len(X_images)

        for w_idx, wl in enumerate(wavelengths):
            # Print current wavelength progress
            print(f"\n[λ {w_idx+1}/{total_wavelengths}] Measuring at {wl}nm...")
            self.laser.set_wavelength(wl, settle=1)
            wl_features = []
            
            for i_idx, img in enumerate(X_images):
                # Terminal Progress Update: Updates on the same line to avoid clutter
                progress = (i_idx + 1) / total_images * 100
                print(f"\r  > Image {i_idx+1}/{total_images} ({progress:.1f}%)", end="", flush=True)

                if isinstance(img, list) or (isinstance(img, np.ndarray) and img.ndim > 1):
                    work_list = img
                else:
                    work_list = [img]
                
                feat_chunks = []
                for row in work_list: 
                    v = np.clip(v_bias + (np.asarray(row) - 0.5) * 2.0 * gain * 2.4, 0.1, 4.9)
                    
                    v_list = v.tolist() if isinstance(v, np.ndarray) else [v]
                    self.bus.set(self.inputs, v_list)
                    
                    feat_chunks.append(self.scope.read_many(avg=1))
                
                wl_features.append(np.array(feat_chunks).flatten())
            
            # Print a newline once the images for this wavelength are done
            print(f"\n  λ {wl}nm Complete.")
            X_stack.append(wl_features)
                
        self.laser.turn_off()
        print("\n[SUCCESS] All measurements completed.")
        return np.stack(X_stack, axis=1)

    def train_elm(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, acc, (X_test, y_test)

    def save_and_plot(self, model, test_data, acc, gain, wavelengths,sample_per_class, params, name="run"):
        X_test, y_test = test_data
        y_pred = model.predict(StandardScaler().fit_transform(X_test))
        
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, cmap='Blues')
        plt.title(
            f"Acc: {acc:.3f} | Gain: {gain} | Inputs: {len(self.inputs)}\n"
            f"$\lambda$: {params['L']} | {wavelengths} | S/C: {sample_per_class}",
            fontsize=10
        )
        
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") 
        plt.savefig(f"results/{name}_{ts}.png")
        plt.show()

    def save_results(self, X_features, y_labels, gain, wavelengths, name="fashion_data"):
        """
        Saves the raw measured features and metadata to a compressed NPZ file.
        """
        os.makedirs("data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/{name}_g{gain}_{ts}.npz"
        
        # Save everything into one package
        np.savez_compressed(
            filename,
            features=X_features.astype(np.float32),
            labels=y_labels,
            gain=gain,
            wavelengths=np.array(wavelengths),
            input_heaters=np.array(self.inputs),
            mesh_heaters=np.array(self.mesh)
        )
        print(f"\n[DATA SAVED] Full dataset saved to: {filename}")
        return filename