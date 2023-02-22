import math


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
        self.expiry = expiry
        self.k = k
        self.num_steps = num_steps
        self.dt = expiry / num_steps
        self.mc_paths = None
