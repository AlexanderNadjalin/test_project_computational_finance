import math
import numpy as np


class RnDisc:
    """

    Class for generating risk-neutral discount factor given short rate and time in year fraction.
    """
    def __init__(self,
                 r: float,
                 dt: float):
        """

        Initializing class.
        :param r: Short rate in decimal form.
        :param dt: Time (year fraction) in decimal form.
        """
        self.short_rate = r
        self.year_frac = dt
        self.df = self.df()

    def df(self) -> float:
        """

        Calculate discount factor.
        :return: Risk-neutral discount factor.
        """
        return math.exp(-self.short_rate * self.year_frac)


class MarketAndOption:
    def __init__(self,
                 s0: float,
                 sigma: float,
                 r: float,
                 option_type: str,
                 expiry: float,
                 k: float,
                 num_steps: int):
        self.s0 = s0
        self.sigma = sigma
        self.r = r
        self.option_type = option_type
        self.expiry = expiry * 365
        self.k = k
        self.num_steps = num_steps
        self.dt = expiry / num_steps
        self.mc_paths = None

        # Run the Monte-Carlo simulation.
        self.mc_sim()

    def mc_sim(self):
        # Pre-simulation setup.
        rand_s = np.random.standard_normal(self.num_steps)
        self.mc_paths = np.zeros_like(rand_s)
        self.mc_paths[0] = self.s0

        # The Euler discretization at time t.
        for t in range(1, self.num_steps):
            self.mc_paths[t] = self.mc_paths[t - 1] * \
                               math.exp((self.r - self.sigma * self.sigma / 2) * self.dt +
                                        self.sigma * math.sqrt(self.dt) * rand_s[t])


if __name__ == '__main__':
    s0 = 100
    sigma = 0.2
    r = 0.05
    expiry = 1
    k = 100
    num_steps = 1000

    d = MarketAndOption(s0=s0,
                        sigma=sigma,
                        r=r,
                        option_type='d',
                        expiry=expiry,
                        k=k,
                        num_steps=num_steps)
    print(d)
