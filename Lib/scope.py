"""
Simple Rigol Scope Interface
============================
"""

from concurrent.futures import ThreadPoolExecutor
import time, re
import numpy as np
import pyvisa

class Rigol_Scopes:
    def __init__(self, channels_scope1, channels_scope2, serial_scope1=None, timeout_ms=3000):
        self.channels1 = list(channels_scope1)
        self.channels2 = list(channels_scope2)
        self.serial_scope1 = serial_scope1
        self.timeout_ms = timeout_ms
        
        # 1. PERSISTENT EXECUTOR: Avoids the overhead of creating threads on every read
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._connect()

    def _connect(self):
        rm = pyvisa.ResourceManager()
        self.rm = rm
        candidates = []

        for addr in rm.list_resources():
            try:
                inst = rm.open_resource(addr)
                inst.timeout = self.timeout_ms
                inst.read_termination = '\n'
                inst.write_termination = '\n'
                idn = inst.query("*IDN?").strip()
                if any(x in idn.upper() for x in ["RIGOL", "HDO"]):
                    candidates.append({
                        "addr": addr, "idn": idn,
                        "serial": self._parse_serial(idn), "inst": inst
                    })
                else:
                    inst.close()
            except Exception:
                pass

        if len(candidates) < 2:
            raise RuntimeError(f"[SCOPE] Need 2 Rigol scopes; found {len(candidates)}")

        # Determine scope1 vs scope2
        if self.serial_scope1:
            scope1_data = next((c for c in candidates if c["serial"] == self.serial_scope1), None)
            if not scope1_data: raise RuntimeError("serial_scope1 not found.")
        else:
            candidates.sort(key=lambda x: x["serial"])
            scope1_data = candidates[0]

        scope2_data = [c for c in candidates if c is not scope1_data][0]
        self.scope1, self.scope2 = scope1_data["inst"], scope2_data["inst"]

        # 2. OPTIMIZED VERTICAL RANGE: 
        # For 0-5V logic, 1V/div with -2.5V offset uses more of the ADC than 2V/div.
        for s, chs in [(self.scope1, self.channels1), (self.scope2, self.channels2)]:
            s.write(":TIMebase:MAIN:SCALe 1e-6")
            for ch in chs:
                s.write(f":CHANnel{ch}:DISPlay ON")
                s.write(f":CHANnel{ch}:SCALe 2.0")  # Tighter scale for better precision
                s.write(f":CHANnel{ch}:OFFSet -6.0") # Centers 0-5V signal on screen
        
        time.sleep(0.1)

    def _parse_serial(self, idn: str) -> str:
        parts = [p.strip() for p in idn.split(',')]
        return parts[2] if len(parts) >= 3 else idn

    def _read_fast(self, scope, ch):
        """Uses direct measurement instead of statistic queries for speed."""
        try:
            # :MEASure:ITEM? VMAX is significantly faster than :STATistic:ITEM?
            return float(scope.query(f':MEASure:ITEM? VMAX,CHANnel{ch}'))
        except:
            return np.nan

    def _read_scope_batch(self, scope, channels, avg):
        out = []
        for ch in channels:
            samples = []
            for _ in range(avg):
                v = self._read_fast(scope, ch)
                if v == v: samples.append(v) # Fast NaN check
            out.append(sum(samples)/len(samples) if samples else np.nan)
        return out

    def read_many(self, avg=1):
        """
        Parallelized readout using a persistent thread pool.
        """
        # Small stabilization delay for the hardware
        time.sleep(0.04) 
        
        # Submit tasks to the pre-existing pool
        f1 = self.executor.submit(self._read_scope_batch, self.scope1, self.channels1, avg)
        f2 = self.executor.submit(self._read_scope_batch, self.scope2, self.channels2, avg)
        
        return np.array(f1.result() + f2.result(), dtype=float)

    def close(self):
        self.executor.shutdown(wait=False)
        for s in (self.scope1, self.scope2):
            try: s.close()
            except: pass
        self.rm.close()