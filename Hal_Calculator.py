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
