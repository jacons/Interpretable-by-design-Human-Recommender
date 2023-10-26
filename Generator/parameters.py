import numpy as np

file_path = dict(
    jobs_lib="../sources/jobs_library.json",
    cities="../sources/all_cities.csv",
    nationality="../sources/languages.csv",
    education="../sources/education.csv",
    language_level="../sources/language_level.csv",

    lang_level_dist=[0.70, 0.20, 0.10],
    certificates_dist=[0.41, 0.28, 0.19, 0.12]
)

curriculum_par = dict(
    size=750,
    path="../outputs/curricula.csv"
)

jobOffer_par = dict(
    size=120,
    path="../outputs/jobOffers.csv"
)

matching_par = dict(
    cities_dist="../sources/cities_distance.csv",
    labels=5,
    weight=np.array([
        12,  # Education
        2,  # City
        15,  # Skills
        6,  # SoftSkills
        3,  # Age
        8,  # Language
        5,  # Certificates
        4,  # Experience
        2,  # Offered_Salary
        1,  # SmartWork
        1,  # Experience abroad
    ], dtype=np.float32),
    noise=(0, 0.2)  # mean and stddev
)
