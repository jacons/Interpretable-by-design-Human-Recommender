import numpy as np

file_paths = dict(
    job2skills_path="../sources/job2skills.csv",
    occupation_path="../sources/occupations.csv",
    skills_path="../sources/skills.csv",
    cities_path="../sources/all_cities.csv",  # All cites used
    languages_level_path="../sources/languages_level.csv",  # All language levels
    languages_path="../sources/languages.csv",  # All language
    education_path="../sources/education.csv",  # Education hierarchy

    min_edu_occupation_path="../sources/min_education.csv",  # Minimal education for isco groups
    skill_synonyms_path="../sources/skills_synonyms.csv",  # Skills synonyms

    lang_level_distribution=[0.65, 0.35],  # Distribution of extra languages that the candidate knows [0-1]
    # certificates_distribution=[0.41, 0.28, 0.19, 0.12] # Distribution of certificates [0-1-2-3]
)

curriculum_par = dict(
    mu=80,
    path="../outputs/curricula.csv"
)

jobOffer_par = dict(
    size=100,
    path="../outputs/jobOffers.csv"
)

matching_par = dict(
    cities_dist="../sources/cities_distance.csv",
    bins=5,

    weight=np.array([
        8,  # Education essential
        4,  # Education optional
        1,  # City
        2,  # Age
        3,  # Experience essential
        1,  # Experience optional
        8,  # Language essential
        5,  # Language optional
        10,  # Competence essential
        7,  # Competence optional
        10,  # Knowledge essential
        7,  # Knowledge optional
    ], dtype=np.float32),
    noise=(0, 0.01),  # mean and stddev

    split_size=(0.33, 0.33),  # Hold-out
    split_seed=841  # Reproducible splitting
)
