from typing import List
import pandas as pd


class IntervalToSource:
    def __init__(self):
        self.starting_points: List[int] = []
        self.sources: List[str] = []

    def append_interval(self, start: int, source: str) -> None:
        self.starting_points.append(start)
        self.sources.append(source)

    def get_source(self, index: int) -> str:
        if len(self.starting_points) == 0:
            raise IndexError("No intervals were set")

        if self.starting_points[0] > index:
            raise IndexError("Index is less then first starting point")

        for i in range(len(self.starting_points) - 1):
            if self.starting_points[i] <= index < self.starting_points[i + 1]:
                return self.sources[i]

    def __str__(self) -> str:
        text = "{ "
        for i in range(len(self.starting_points) - 1):
            text += f"[{self.starting_points[i]}, {self.starting_points[i + 1]})"
            text += f" -> \"{self.sources[i]}\"\n  "
        text += f"[{self.starting_points[len(self.starting_points) - 1]}, ∞)"
        text += f" -> \"{self.sources[len(self.starting_points) - 1]}\"" + " }\n"
        return text

    def to_csv(self, file: str):
        df = pd.DataFrame()
        df['Starting Point'] = self.starting_points
        df['Source'] = self.sources
        df.to_csv(file)

    @staticmethod
    def read_csv(file: str) -> "IntervalToSource":
        df = pd.read_csv(file)
        i2s = IntervalToSource()
        i2s.sources = df['Source'].to_list()
        i2s.starting_points = df['Starting Point'].to_list()
        return i2s