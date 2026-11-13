import sys; sys.path.insert(0, ".")
import numpy as np
from aqi import pm25_subindex, pm25_category
import eval_config as C

# Ground truth: values recalculated from the official CPCB AQI-Calculator.xls (cell D10 formula)
SHEET = {0: 0, 15: 25, 30: 50, 30.5: 50.8333, 45: 75, 60: 100, 75: 150, 89.99: 199.9667, 90: 200,
         90.01: 200.0333, 105: 250, 120: 300, 121: 300.7692, 185: 350, 250: 400, 251: 400.7692, 300: 438.4615,
         34: 56.6667}   # 34 is the sheet's own worked example

def test_all():
    for c, want in SHEET.items():
        assert abs(float(pm25_subindex(c)) - want) < 1e-3, (c, pm25_subindex(c), want)
    assert float(pm25_subindex(C.SPIKE_THRESHOLD)) == 200            # threshold = Moderate/Poor boundary
    assert pm25_category(90) == "Moderate" and pm25_category(90.01) == "Poor"
    assert pm25_category(30) == "Good" and pm25_category(300) == "Severe"
    assert pm25_category(np.nan) is None

if __name__ == "__main__":
    test_all(); print("PASS aqi")
