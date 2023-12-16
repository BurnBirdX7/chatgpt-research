
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

# Wiki Articles
page_names: list[str] = [
    "John_F._Kennedy",
    "Winston_Churchill",
    "Joseph_Stalin",
    "Marie_Antoinette",
    "Henry_Ford",
    "Brigham_Young",
    "Charles_Cornwallis",
    "Jack_the_Ripper",
    "Muhammad_Ali",
    "Douglas_MacArthur",
    "Sun_Tzu",
    "Frank_Sinatra",
    "Albert_Einstein",
    "George_Washington",
    "Pope_John_Paul_II",
    "Vince_Lombardi",
    "Abraham_Lincoln",
    "Benjamin_Franklin",
    "Thomas_Edison",
    "Elvis_Presley",
    "Napoleon",
    "Mark_Twain",
    "Ronald_Reagan",
    "Socrates",
    "Martin_Luther_King_Jr",
    "Oliver_Wendell_Holmes_Sr.",
    "Oliver_Wendell_Holmes_Jr.",
    "Oliver_Wendell_Holmes_(archivist)",
    "Wendell_Holmes_(actor)",
    "Confucius",
    "Julius_Caesar",
    "George_Foreman",
    "J._Robert_Oppenheimer",
    "Alexander_the_Great",
    "Steve_Jobs",
    "Dwight_D._Eisenhower",
    "Mahatma_Gandhi",
    "Dolly_Parton",
    "Bill_Clinton",
    "Bernie_Madoff",
    "Pablo_Picasso",
    "Stephen_Hawking",
    "Donald_Trump",
    "Vince_Lombardi",
    "Galileo_Galilei",
    "Nelson_Mandela",
    "Al_Capone",
    "Patrick_Henry",
    "Joey_Chestnut",
    "Jonas_Salk",
    "Friedrich_Nietzsche",
    "George_Washington",
    "Neil_Armstrong",
    "John_Wilkes_Booth",
    "Maximilien_Robespierre",
    "Karl_Marx",
    "Salvador_Dali",
    "John_Lennon",
    "David_Lynch",
    "Andy_Warhol",
    "Sigmund_Freud",
    "Oprah_Winfrey",
    "Benedict_Arnold",
    "History_of_France",
    "Military_history_of_France",
    "History_of_French",
    "Economic_history_of_France",
    "Political_history_of_France",
    "Timeline_of_French_history",
    "History_of_education_in_France",
    "History_of_French_Guiana",
    "LGBT_history_in_France",
    "History_of_the_Jews_in_France",
    "World_War_II",
    "World_War_II_casualties",
    "Allies_of_World_War_II",
    "World_War_II_by_country",
    "Aftermath_of_World_War_II",
    "List_of_timelines_of_World_War_II",
    "Pacific_War",
    "War_crimes_in_World_War_II",
    "European_theatre_of_World_War_II",
    "Organelle",
    "Cell_nucleus",
    "Cell_biology",
    "Cell_membrane",
    "Cell_(biology)",
    "Cell_physiology",
    "Vacuole",
    "Cytoskeleton",
    "Outline_of_cell_biology",
    "Cytoplasm",
    "FIFA_World_Cup",
    "Unofficial_Football_World_Championships",
    "Premier_League",
    "List_of_English_football_champions",
    "List_of_Premier_League_clubs",
    "Premier_League_records_and_statistics",
    "Association_football",
    "FIFA",
    "UEFA_European_Championship",
    "UEFA_Champions_League",
    "UEFA",
    "List_of_Big_Ten_Conference_football_champions",
    "National_park",
    "List_of_national_parks",
    "List_of_national_parks_of_the_United_States",
    "National_parks_of_the_United_Kingdom",
    "National_parks_of_New_Zealand",
    "Sequoia_National_Park",
    "History",
    "Jewish_history",
    "Timelines_of_world_history",
    "History_of_Europe",
    "Ancient_history",
    "Human_history",
    "Cultural_history",
    "Post-classical_history",
    "Modern_era",
    "Paleolithic",
    "Mesolithic",
    "Neolithic",
    "Chalcolithic"
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
