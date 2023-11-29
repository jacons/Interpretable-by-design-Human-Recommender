from enum import Enum

import numpy as np

file_paths = dict(
    job2skills_path="../sources/job2skills.csv",
    occupation_path="../sources/occupations.csv",
    skills_path="../sources/skills.csv",
    cities_path="../sources/all_cities.csv",  # All cites used
    languages_level_path="../sources/languages_level.csv",  # All language levels
    languages_path="../sources/languages.csv",  # All language
    education_path="../sources/education.csv",  # Education hierarchy
    cities_dist="../sources/cities_distance.csv",
    min_max_edu_occupation_path="../sources/min_max_education.csv",  # Minimal/Maximal education for isco groups
    skill_synonyms_path="../sources/skills_synonyms.csv",  # Skills synonyms

    opt_lang_distribution=[0.65, 0.35],  # Distribution of extra languages that the candidate knows [0-1]
)
job_graph_par = dict(  # parameter for "job-graph class"
    force_build=False,  # If true, create always the graph from scratch
    cache_path="../outputs"  # path of a graph cache (json file cache)
)
match_score_par = dict(  # parameter for "matching score class"
    path="../outputs/scores"
)
curriculum_par = dict(  # parameters for "curricula generator"
    mu=80,  # average of curricula for job-offer
)
jobOffer_par = dict(  # parameter for "job-offer generator"
    size=200,  # number of jobs-offer
)
output_dir = dict(  # parameter for "saving" the synthetic data
    path="../outputs"
)
matching_par = dict(  # parameter for label generator
    bins=6,
    noise=(0, 0.05),  # mean and stddev
    split_size=(0.33, 0.33),  # Hold-out
    split_seed=841,  # Reproducible splitting
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
    ], dtype=np.float32)
)


class RelationNode(Enum):
    ALL = "All"
    ES = "essential"
    OP = "optional"


class TypeNode(Enum):
    OC = "occupation"
    KN = "knowledge"
    SK = "skill/competence"
    ALL = "All"


class EducationLevel(Enum):
    Null = 0
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


class Language:

    def __init__(self, name: str = "-", level: str = "-"):
        self.name = name
        self.level = level
