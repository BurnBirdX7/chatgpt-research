# Misc
model_name: str = "roberta-large"
faiss_use_gpu: bool = False
show_plot: bool = True
threshold = 0.8

# Files:
artifacts_folder = "artifacts"
embeddings_file: str = "roberta-large.csv"
mapping_file: str = "mapping-large.csv"
index_file: str = "large.index"
centroid_file: str = "centroid.npy"
temp_index_file: str = "temp.index"
temp_mapping_file: str = "temp-mapping.csv"
source_index_path = None  # None field will be loaded from environment

# Wiki Articles
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
    "Blue_Hawaii",
]

unrelated_page_names: list[str] = [
    "Stand-up_comedy",
    "FANUC",
    "Francis_Augustus_Cox",
    "(We_All_Are)_Looking_for_Home",
    "Computer_science",
    "3D_printing",
    "Thermoplastic",
]

unrelated_page_names_2: list[str] = [
    "Knut_Storberget",
    "One_of_Those_Nights_(Juicy_J_song)",
    "Politically_exposed_person",
    "Eulaema",
    "Struell_Wells",
    "Pollinator",
    "Sir_Alexander_Fleming_College",
    "Amy_Hughes_(artist)",
    "Jing_Lusi",
    "Recurring_Saturday_Night_Live_characters_and_sketches_introduced_2007-08",
    "Trout_Creek_Hill",
    "Shaynna_Blaze",
    "Leckhampstead,_Buckinghamshire",
    "Mu_Cassiopeiae",
    "Dave_Karnes",
    "Akron_Goodyear_Wingfoots",
    "Australian_cricket_team_in_India_in_2000-01",
    "Sergio_Hernandez_(basketball)",
    "Phil_Joanou",
    "Epiphany_Apostolic_College",
    "WGN-TV",
    "Jacob_Josefson",
    "We_Connect",
    "Tiare_Aguilera_Hey",
    "Apna_Bombay_Talkies",
    "Battle_of_Cravant",
    "So_This_Is_Paris_(1926_film)",
]
