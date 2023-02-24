import math
import numpy as np
import matplotlib.pyplot as plt


class Df:
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
    """

    Class holding market parameters, simulation parameters and option information.
    """
    def __init__(self,
                 s0: float,
                 d: float,
                 sigma: float,
                 r: float,
                 option_type: str,
                 expiry: float,
                 k: float,
                 num_steps: int,
                 num_sims: int):
        # Underlying price at time 0.
        self.s0 = s0
        # Constant dividend rate (decimal form).
        self.d = d
        # Constant volatility (decimal form).
        self.sigma = sigma
        # Constant short rate (decimal form).
        self.r = r
        self.option_type = option_type
        # Time to expiry expressed as a year fraction.
        self.expiry = expiry * 365
        # Option strike.
        self.k = k
        # Number of steps in each Monte-Carlo path.
        self.num_steps = num_steps
        # Number of Monte-Carlo paths.
        self.num_sims = num_sims
        # Each step size in a Monte-Carlo path.
        self.dt = expiry / num_steps
        # Holder for all Monte-Carlo paths.
        self.mc_paths = None

        # Run the Monte-Carlo simulation.
        self.mc_sim()

        # Holder for plot.
        self.plot = None

    def mc_sim(self):
        """

        Monte-Carlo simulation.
        """
        # Pre-simulation setup.
        rand_s = np.random.standard_normal((self.num_steps, self.num_sims))
        self.mc_paths = np.zeros_like(rand_s)
        # Set s0 as first value.
        self.mc_paths[0] = self.s0

        # The Euler discretization for all time steps.
        for t in range(1, self.num_steps):
            self.mc_paths[t] = self.mc_paths[t - 1] * \
                               np.exp((self.r - self.d - self.sigma ** 2 / 2) * self.dt +
                                      self.sigma * math.sqrt(self.dt) * rand_s[t])

    def plot_paths(self):
        plt.figure(figsize=(15, 12))
        plt.plot(self.mc_paths)
        plt.legend()
        plt.title('Euler discretization scheme')
        plt.ylabel('S')
        plt.grid(True)
        plt.axis('tight')

        self.plot = plt
        self.plot.show()


if __name__ == '__main__':
    s0 = 100
    d = 0.0
    sigma = 0.2
    r = 0.05
    expiry = 1
    k = 100
    num_steps = 100
    num_sims = 100

    d = MarketAndOption(s0=s0,
                        d=d,
                        sigma=sigma,
                        r=r,
                        option_type='d',
                        expiry=expiry,
                        k=k,
                        num_steps=num_steps,
                        num_sims=num_sims)

    d.plot_paths()

