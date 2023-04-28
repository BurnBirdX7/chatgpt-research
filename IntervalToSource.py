from typing import List
import pandas as pd  # type: ignore


class IntervalToSource:
    """
    Helps to create association between wide range of values and their source

    Every new range represented by its upper bound and associated value

    [0, x) -> v_x
    [x, y) -> v_y
    [y, z) -> v_z

    """

    def __init__(self, lowest_bound: int = 0):
        self.lowest_bound: int = lowest_bound
        self.upper_limits: List[int] = []
        self.sources: List[str] = []

    def append_interval(self, upper_limit: int, source: str) -> None:
        """
        Append new range

          [0, 10) -> google.com
          [10, 16) -> ya.ru

        append_interval(21, 'bing.com')

          [0, 10) -> google.com
          [10, 16) -> ya.ru
          [16, 21) -> bing.com
        """
        self.upper_limits.append(upper_limit)
        self.sources.append(source)

    def get_source(self, index: int) -> str:
        if len(self.upper_limits) == 0:
            raise IndexError("No intervals were set")

        if self.lowest_bound > index:
            raise IndexError("Index is less then first range's lower bound")

        if self.upper_limits[-1] <= index:
            raise IndexError("Index is greater then last range's upper bound")

        for ub, src in zip(self.upper_limits, self.sources):
            if index < ub:
                return src

        raise RuntimeError("Unexpected error: unreachable")

    def __str__(self) -> str:
        text = "{ "
        text += f"[{self.lowest_bound}, {self.upper_limits[0]})"
        text += f' -> "{self.sources[0]}"'
        for i in range(len(self.upper_limits) - 1):
            text += f"\n[{self.upper_limits[i]}, {self.upper_limits[i + 1]})"
            text += f' -> "{self.sources[i + 1]}"'
        text += "}\n\n"
        return text

    def to_csv(self, file: str) -> None:
        df = pd.DataFrame()
        df["Upper Bound"] = self.upper_limits
        df["Source"] = self.sources
        df.to_csv(file, index=False)

    @staticmethod
    def read_csv(file: str) -> "IntervalToSource":
        df = pd.read_csv(file)
        i2s = IntervalToSource()
        i2s.upper_limits = df["Upper Bound"].to_list()
        i2s.sources = df["Source"].to_list()
        return i2s
