class HalCalculator:
    """
    Incrementally calculates the average of squared deviations from the mean.
    That is, Hal(x) = (1/N) * sum((x_i - mean)^2), which is effectively the variance.
    """

    def __init__(self):
        # n     : count of values so far
        # mean  : current mean of all values
        # m2    : sum of squared deviations from the mean
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> float:
        """
        Add a new value x and update the running statistics.
        Returns the current Hal(x) (variance) right after this addition.
        """
        self.n += 1
        # Welford's online update:
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

        return self.item()

    def item(self) -> float:
        """
        Returns the current value of Hal(x).
        If fewer than 2 values have been seen, returns 0.
        """
        if self.n < 2:
            return 0.0
        return self.m2 / self.n



class HalCalculatorTensor:
    """
    Incrementally calculates the average of squared deviations from the mean
    for a tensor of any shape. Each new tensor must have the same shape.
    
    Hal(x) is computed elementwise:
        Hal(x) = (1 / n) * sum((x_i - mean)^2)
    which is effectively the population variance at each element.
    """

    def __init__(self, shape, timestep_range: int = 200):
        """
        Initialize with a specific tensor shape.
        
        :param shape: tuple describing the shape of each input tensor.
        """
        self.n = 0
        self.mean = torch.zeros(shape)  # Running mean (same shape)
        self.m2   = torch.zeros(shape)  # Sum of squared deviations (same shape)
        self.timestep_range = timestep_range    

    def add(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add a new tensor and update the running statistics.
        
        :param x: A new tensor of the same shape as specified in __init__.
        :return:  The current Hal(x) (elementwise average of squared deviations).
        """
        if self.n == 0:
            # First tensor: mean = x, m2 stays zero
            self.mean = x.clone()
            self.n = 1
            return self.item()  # This will be all zeros if n=1

        self.n += 1
        # Welford's online update
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

        return self.item()

    def item(self) -> torch.Tensor:
        """
        Return the current elementwise average of squared deviations
        (which is effectively the population variance for each element).
        
        :return: A tensor of the same shape, containing Hal(x) values.
        """
        if self.n < 2:
            # With fewer than 2 values, variance is zero
            return torch.zeros_like(self.mean)
        return self.m2 / self.timestep_range
