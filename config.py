model_name: str = "roberta-large"
embeddings_file: str = "roberta-large.csv"
ranges_file: str = "ranges.csv"
index_file: str = "large.index"
faiss_use_gpu: bool = False
show_plot: bool = True

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

unrelated_page_names: list[str] = [
    "Stand-up_comedy",
    "FANUC",
    "Francis_Augustus_Cox",
    "(We_All_Are)_Looking_for_Home",
    "Computer_science",
    "3D_printing",
    "Thermoplastic"
]
