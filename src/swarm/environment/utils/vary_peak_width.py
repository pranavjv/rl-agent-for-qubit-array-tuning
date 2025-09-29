import numpy as np

class VaryPeakWidth:
    def __init__(self, peak_width_0: float, alpha: float = 0.01):
        self.peak_width_0 = peak_width_0
        self.alpha = alpha

    def linearly_vary_peak_width(self, v_x, v_y) -> float:
        magnitude = np.linalg.norm(np.array([v_x, v_y]))
        peak_width = self.peak_width_0 + self.alpha * magnitude

        return peak_width







