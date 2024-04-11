from typing import List, Tuple
import pandas as pd  # type: ignore


class SourceMapping:
    """
    Helps to create association between wide range of values and their source

    Every new range represented by its upper bound and associated value

    [0, x) -> v_x
    [x, y) -> v_y
    [y, z) -> v_z
    """

    def __init__(self: "SourceMapping") -> None:
        self.lowest_bound: int = 0
        self.upper_limits: List[int] = []
        self.sources: List[str] = []

    @property
    def highest_bound(self):
        return (
            self.upper_limits[-1] if len(self.upper_limits) > 0 else self.lowest_bound
        )

    def append_interval(self, length: int, source: str) -> None:
        """
        Appends new interval of length `length` associated with source `source`

          [0, 10) -> google.com
          [10, 16) -> ya.ru

        append_interval(5, 'google.com')

          [0, 10) -> google.com
          [10, 16) -> ya.ru
          [16, 21) -> google.com

        append_interval(3, 'google.com'), again

          [0, 10) -> google.com
          [10, 16) -> ya.ru
          [16, 24) -> google.com
        """
        if len(self.sources) > 0 and self.sources[-1] == source:
            self.upper_limits[-1] += length
            return

        self.upper_limits.append(self.highest_bound + length)
        self.sources.append(source)

    def __assert_index(self, index: int) -> None:
        if len(self.upper_limits) == 0:
            raise IndexError("No intervals were set")

        if self.lowest_bound > index:
            raise IndexError("Index is less then first range's lower bound")

        if self.upper_limits[-1] <= index:
            raise IndexError("Index is greater then last range's upper bound")

    def get_source_and_interval(self, index: int) -> Tuple[str, int, int]:
        """
        Returns interval and source associated with the index

        Given mapping `m`:
          [0, 10) -> google.com
          [10, 16) -> ya.ru
          [16, 21) -> bing.com

        Call
          m.get_source_and_interval(12)

        will return
          ("ya.ru", 10, 16)
        """
        self.__assert_index(index)

        prev_bound = self.lowest_bound
        for ub, src in zip(self.upper_limits, self.sources):
            if index < ub:
                return src, prev_bound, ub
            prev_bound = ub

        raise RuntimeError("Unexpected error: unreachable")

    def get_source(self, index: int) -> str:
        """
        Returns source associated with the index

        Given mapping `m`:
          [0, 10) -> google.com
          [10, 16) -> ya.ru
          [16, 21) -> bing.com

        Call
          m.get_source(12)

        will return
          "ya.ru"
        """
        return self.get_source_and_interval(index)[0]

    def get_interval(self, index: int) -> Tuple[int, int]:
        """
        Returns interval associated with the index

        Given mapping `m`:
          [0, 10) -> google.com
          [10, 16) -> ya.ru
          [16, 21) -> bing.com

        Call
          m.get_interval(12)

        will return
          (10, 16)
        """
        t = self.get_source_and_interval(index)
        return t[1], t[2]

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
    def read_csv(file: str) -> "SourceMapping":
        df = pd.read_csv(file)
        i2s = SourceMapping()
        i2s.upper_limits = df["Upper Bound"].to_list()
        i2s.sources = df["Source"].to_list()
        return i2s
