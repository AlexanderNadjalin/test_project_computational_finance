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
                 div: float,
                 sigma: float,
                 r: float,
                 expiry: float,
                 k: float,
                 num_steps: int,
                 num_sims: int):
        """

        Initializing class.
        @param s0: Underlying stock price at time t0.
        @param div: Dividends (as a continuous yield).
        @param sigma: Constant volatility (in decimal form).
        @param r: Constant short rate (in decimal form).
        @param expiry: Time to expiry (in years).
        @param k: Option strike level.
        @param num_steps: Number of steps in each Monte-Carlo simulation.
        @param num_sims:Number om Monte-Carlo simulations.
        """
        self.s0 = s0
        self.div = div
        self.sigma = sigma
        self.r = r
        self.expiry = expiry
        self.k = k
        self.num_steps = num_steps
        self.num_sims = num_sims
        self.dt = expiry / num_steps
        self.mc_paths = None

        # Run the Monte-Carlo simulation.
        self.mc_sim()

        # Holder for plot.
        self.plot = None

    def mc_sim(self) -> None:
        """

        Monte-Carlo simulation.
        @return: None.
        """
        # Pre-simulation setup.
        rand_s = np.random.standard_normal((self.num_steps, self.num_sims))
        self.mc_paths = np.zeros_like(rand_s)
        # Set s0 as first value.
        self.mc_paths[0] = self.s0

        # The Euler discretization for all time steps.
        for t in range(1, self.num_steps):
            self.mc_paths[t] = self.mc_paths[t - 1] * \
                               np.exp((self.r - self.div - self.sigma ** 2 / 2) * self.dt +
                                      self.sigma * math.sqrt(self.dt) * rand_s[t])

    def plot_paths(self) -> None:
        """

        Plot all paths.
        @return: None.
        """
        plt.figure(figsize=(15, 12))
        plt.plot(self.mc_paths)
        plt.title('Euler discretization scheme')
        plt.ylabel('S')
        plt.xlabel('delta_t')
        plt.grid(True)
        plt.axis('tight')

        self.plot = plt
        self.plot.show()


class VanillaOption:
    """

    Class for calculating the price of a European plain vanilla call or put option.
    """
    def __init__(self,
                 market_and_option: MarketAndOption,
                 is_call: bool) -> None:
        """

        Calculate the price of a European plain vanilla call or put option.
        @param market_and_option: MarketAndOption object.
        @param is_call: Boolean.
        """
        self.mo = market_and_option
        self.is_call = is_call
        self.payoff = None
        if self.is_call:
            self.payoff = np.maximum(self.mo.mc_paths[-1] - self.mo.k, 0)
        else:
            self.payoff = np.maximum(self.mo.k - self.mo.mc_paths[-1], 0)
        df = Df(self.mo.r, dt=self.mo.expiry)
        self.price = df.df * self.payoff.mean()


class Lookback:
    """

    Class for calculating price of a European lookback call/put option with fixed/floating strike.
    """
    def __init__(self,
                 market_and_option: MarketAndOption,
                 is_call: bool,
                 is_floating_strike: bool) -> None:
        """

        Calculate the price of a European lookback call/put option with fixed/floating strike.
        @param market_and_option: MarketAndOption object.
        @param is_call: Boolean.
        @param is_floating_strike: Boolean.
        """

        self.mo = market_and_option
        self.is_call = is_call
        self.is_floating_strike = is_floating_strike
        self.payoff = None

        # Lookback values for payoff function.
        s_t = self.mo.mc_paths[-1]
        s_min = self.mo.mc_paths.min()
        s_max = self.mo.mc_paths.max()

        # Floating strike.
        if self.is_floating_strike:
            # Call option.
            if self.is_call:
                self.payoff = np.maximum(s_max - s_t, 0)
            # Put option.
            else:
                self.payoff = np.maximum(s_t - s_min, 0)
        # Fixed strike.
        else:
            # Call option.
            if self.is_call:
                self.payoff = np.maximum(s_max - self.mo.k, 0)
            # Put option.
            else:
                self.payoff = np.maximum(self.mo.k - s_min, 0)
        df = Df(self.mo.r, dt=self.mo.expiry)
        self.price = df.df * self.payoff.mean()


class Asian:
    """

     Class for calculating price of a European asian call/put option with fixed/floating strike.
    """
    def __init__(self,
                 market_and_option: MarketAndOption,
                 is_call: bool,
                 is_floating_strike: bool) -> None:
        """

        Calculate the price of a European asian call/put option with fixed/floating strike.
        @param market_and_option: MarketAndOption object.
        @param is_call: Boolean.
        @param is_floating_strike: Boolean.
        """

        self.mo = market_and_option
        self.is_call = is_call
        self.is_floating_strike = is_floating_strike
        self.payoff = None

        # Asian values for payoff function.
        s_t = self.mo.mc_paths[-1]
        s_mean = self.mo.mc_paths.mean()

        # Floating strike.
        if self.is_floating_strike:
            # Call option.
            if self.is_call:
                self.payoff = np.maximum(s_mean - s_t, 0)
            # Put option.
            else:
                self.payoff = np.maximum(s_t - s_mean, 0)
        # Fixed strike.
        else:
            # Call option.
            if self.is_call:
                self.payoff = np.maximum(s_mean - self.mo.k, 0)
            # Put option.
            else:
                self.payoff = np.maximum(self.mo.k - s_mean, 0)
        df = Df(self.mo.r, dt=self.mo.expiry)
        self.price = df.df * self.payoff.mean()


if __name__ == '__main__':
    s0 = 36.0
    div = 0.0
    sigma = 0.2
    r = 0.06
    expiry = 1
    k = 40.0
    num_steps = 100
    num_sims = 50000

    d = MarketAndOption(s0=s0,
                        div=div,
                        sigma=sigma,
                        r=r,
                        expiry=expiry,
                        k=k,
                        num_steps=num_steps,
                        num_sims=num_sims)

    # d.plot_paths()
    vanilla = VanillaOption(market_and_option=d,
                            is_call=False)
    # print(vanilla.price)

    lookback = Lookback(market_and_option=d,
                        is_call=True,
                        is_floating_strike=True)
    print(lookback.price)
    lookback = Lookback(market_and_option=d,
                        is_call=False,
                        is_floating_strike=True)
    print(lookback.price)

    asian = Asian(market_and_option=d,
                  is_call=True,
                  is_floating_strike=True)
    print(asian.price)
    asian = Asian(market_and_option=d,
                  is_call=False,
                  is_floating_strike=True)
    print(asian.price)
