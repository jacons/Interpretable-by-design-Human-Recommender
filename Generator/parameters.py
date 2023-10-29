import numpy as np

file_path = dict(
    jobs_lib="../sources/jobs_library.json",
    cities="../sources/all_cities.csv",
    nationality="../sources/languages.csv",
    education="../sources/education.csv",
    language_level="../sources/language_level.csv",

    lang_level_dist=[0.35, 0.45, 0.20],
    certificates_dist=[0.41, 0.28, 0.19, 0.12]
)

curriculum_par = dict(
    size=300,
    path="../outputs/curricula.csv"
)

jobOffer_par = dict(
    size=50,
    path="../outputs/jobOffers.csv"
)

matching_par = dict(
    cities_dist="../sources/cities_distance.csv",
    bins=5,

    weight=np.array([
        12,  # Education
        2,  # City
        20,  # Skills
        10,  # SoftSkills
        3,  # Age
        8,  # Language
        5,  # Certificates
        4,  # Experience
        2,  # Offered_Salary
        1,  # SmartWork
        1,  # Experience abroad
    ], dtype=np.float32),
    noise=(0, 0.1),  # mean and stddev

    split_size=(0.33, 0.33),  # Hold-out
    split_seed=841  # Reproducible splitting
)
