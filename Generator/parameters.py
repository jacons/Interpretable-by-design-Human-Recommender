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
    occupation_synonyms_path="../sources/occupation_synonyms.csv",  # Occupation synonyms
    skill_synonyms_path="../sources/skills_synonyms.csv.csv",  # Skills synonyms

    lang_level_distribution=[0.65, 0.35],  # Distribution of extra languages that the candidate knows [0-1]
    # certificates_distribution=[0.41, 0.28, 0.19, 0.12] # Distribution of certificates [0-1-2-3]
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
