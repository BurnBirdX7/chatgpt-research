from dataclasses import dataclass

from BaseConfig import BaseConfig, DefaultValue


@dataclass
class WikiConfig(BaseConfig):
    target_pages: list[str]

    @staticmethod
    def get_elvis_config() -> "WikiConfig":
        page_names: list[str] = [
            "Elvis_Presley",
            "List_of_songs_recorded_by_Elvis_Presley_on_the_Sun_label",
            "Cultural_impact_of_Elvis_Presley",
            "Cultural_depictions_of_Elvis_Presley",
            "Elvis_has_left_the_building",
            "Elvis_Presley_on_film_and_television",
            "Love_Me_Tender_(film)",
            "Memphis,_Tennessee",
            "Heartbreak_Hotel",
            "Jailhouse_Rock_(film)",
            "Blue_Hawaii"
        ]

        return WikiConfig(page_names)
