import numpy as np

class OU_process:
    def __init__(self, sig, tc, dt, num_points):
        self.sig = sig
        self.tc = tc
        self.num_points = num_points
        self.dt = dt
        self.x = self.sig * np.random.normal(0,1)
        self.a = (np.exp(-self.dt/self.tc))
        self.b = np.sqrt(1-np.exp(-2*self.dt/self.tc))*self.sig

    def next_val(self):
        n = np.random.normal(0,1)
        self.x = self.x*self.a +self.b*n
        return self.x
    
    def __call__(self):
        vals = np.zeros(self.num_points)
        for i in range(0, self.num_points):
            vals[i] = self.next_val()
        return vals


class FlickerNoise:
    """
    Implementation of 1/f noise (flicker noise) using the Voss-McCartney algorithm.
    This generates noise with a power spectral density proportional to 1/f^alpha.
    
    Parameters:
    - amplitude: RMS amplitude of the noise
    - alpha: Power law exponent (typically 1.0 for flicker noise)
    - dt: Time step
    - num_points: Number of points to generate
    - f_min: Minimum frequency (default: 1/num_points)
    - f_max: Maximum frequency (default: 1/(2*dt))
    """
    def __init__(self, amplitude, alpha=1.0, dt=1.0, num_points=1000, f_min=None, f_max=None):
        self.amplitude = amplitude
        self.alpha = alpha
        self.dt = dt
        self.num_points = num_points
        self.f_min = f_min if f_min is not None else 1.0 / num_points
        self.f_max = f_max if f_max is not None else 1.0 / (2.0 * dt)
        
        # Generate the noise sequence
        self._generate_noise()
        
    def _generate_noise(self):
        """Generate 1/f noise using Voss-McCartney algorithm."""
        # Initialize array
        self.noise = np.zeros(self.num_points)
        
        # Number of octaves
        n_octaves = int(np.log2(self.f_max / self.f_min))
        
        # Generate noise for each octave
        for octave in range(n_octaves):
            # Frequency for this octave
            f = self.f_min * (2 ** octave)
            
            # Period for this frequency
            period = int(1.0 / (f * self.dt))
            
            if period > 0 and period < self.num_points:
                # Generate random values for this frequency component
                values = np.random.normal(0, 1, (self.num_points // period + 1))
                
                # Interpolate to full length
                indices = np.arange(0, self.num_points, period)
                for i, idx in enumerate(indices):
                    if idx < self.num_points:
                        self.noise[idx] += values[i] / (f ** (self.alpha / 2.0))
        
        # Normalize to desired amplitude
        current_rms = np.sqrt(np.mean(self.noise**2))
        if current_rms > 0:
            self.noise = self.noise * (self.amplitude / current_rms)
        
        # Reset index for iteration
        self.current_index = 0
    
    def next_val(self):
        """Get next value from the noise sequence."""
        if self.current_index >= self.num_points:
            # Regenerate noise when we reach the end
            self._generate_noise()
        
        val = self.noise[self.current_index]
        self.current_index += 1
        return val
    
    def __call__(self):
        """Return a batch of noise values."""
        vals = np.zeros(self.num_points)
        for i in range(self.num_points):
            vals[i] = self.next_val()
        return vals
    
    def get_sequence(self, length=None):
        """Get a sequence of noise values of specified length."""
        if length is None:
            length = self.num_points
        
        sequence = np.zeros(length)
        for i in range(length):
            sequence[i] = self.next_val()
        return sequence
    


