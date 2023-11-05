import csv

import pandas as pd
from geopy.distance import geodesic
from pandas import DataFrame
from unidecode import unidecode


def build_cities_distance(raw_sources: str, output_dir: str, sub_sample: float = 0.1) -> DataFrame:
    print("Loading source data...", end="")
    df_geo = pd.read_json(raw_sources + "/italy_geo.json")
    df_cities = pd.read_json(raw_sources + "/italy_cities.json")

    df = df_cities.merge(df_geo[["istat", "lng", "lat"]], on="istat")
    print("Completed")

    print("Data cleaning...", end="")
    # Drop useless columns
    df = df.drop(["istat"], axis=1).dropna().drop(df[df["comune"] == ""].index)
    # Remove special character like à,è ecc...
    df["comune"] = df["comune"].apply(lambda name: unidecode(name))
    # Reduce the dimension of the file and combine the cities
    df = df.sample(frac=sub_sample)

    df["P"] = df["num_residenti"] / df["num_residenti"].sum()
    df[["comune", "num_residenti", "P"]].to_csv(output_dir + "/all_cities.csv", index=False)
    df.drop(["num_residenti"], axis=1, inplace=True)
    print("Completed")

    print("Merging...", end="")
    df = df.merge(df, how='cross')
    # Remove the cities with the same name
    df.drop(df[df["comune_x"] == df["comune_y"]].index, axis=0, inplace=True)
    df = df.rename(columns={"comune_x": "A", "comune_y": "B"})
    print("Completed")

    print("Optimizing...", end="")
    # Remove all duplicate rows Es. A->B and B->A
    df.sort_values(["A", "B"], inplace=True)
    df['temp'] = df.apply(lambda row: ''.join(sorted([row['A'], row['B']])), axis=1)
    df = df.drop_duplicates(subset='temp').drop(columns=['temp'])
    print("Completed")

    print("Performing distances...", end="")
    # Calculate the distance between the cities
    df["D"] = df.apply(lambda x: geodesic((x["lat_x"], x["lng_x"]), (x["lat_y"], x["lng_y"])).km, axis=1)
    print("Completed")

    print("Formatting...", end="")
    # Casting and sorting
    df = df[["A", "B", "D"]].astype({"A": "string", "B": "string", "D": "int"})
    df.sort_values(["A", "B", "D"], inplace=True)
    df.to_csv(output_dir + "/cities_distance.csv", index=False)
    print("Completed")

    return df


def prepare_datasets(raw_sources: str, output_dir: str):
    SKILL_PATH = "http://data.europa.eu/esco/skill/"
    OCCUPATION_PATH = "http://data.europa.eu/esco/occupation/"

    print("Preprocessing Occupation and Skill...", end="")
    occupation = pd.read_csv(raw_sources + "/occupations_en.csv")
    occ2skills = pd.read_csv(raw_sources + "/occupationSkillRelations_en.csv")
    skills = pd.read_csv(raw_sources + "/skills_en.csv")

    # ---------------------------------------------------------------------
    occupation = occupation[["conceptUri", "preferredLabel", "altLabels", "code"]].astype(
        dtype={"conceptUri": "string", "preferredLabel": "string", "code": "string",
               "altLabels": "string"})
    # remove the link (to simply the notation)
    occupation["conceptUri"] = occupation["conceptUri"].str.replace(OCCUPATION_PATH, "")
    occupation.rename(
        columns={"conceptUri": "id_occupation", "preferredLabel": "occupation",
                 "code": "group", "altLabels": "synonyms"}, inplace=True)

    occupation["synonyms"] = occupation["synonyms"].str.split("\n")
    occupation["synonyms"] = occupation["synonyms"].fillna("[]")

    occupation_synonyms = []
    for row in occupation.itertuples():
        for syn in row[3]:
            occupation_synonyms.append(dict(label=syn, id_occupation=row[1]))
    occupation_synonyms = pd.DataFrame(occupation_synonyms)
    occupation.dropna(inplace=True)

    # ---------------------------------------------------------------------
    occ2skills = occ2skills[["occupationUri", "relationType", "skillUri"]].astype(
        dtype={"occupationUri": "string", "relationType": "string", "skillUri": "string"})

    # remove the link (to simply the notation)
    occ2skills["occupationUri"] = occ2skills["occupationUri"].str.replace(OCCUPATION_PATH, "")
    occ2skills["skillUri"] = occ2skills["skillUri"].str.replace(SKILL_PATH, "")

    occ2skills.rename(
        columns={"occupationUri": "id_occupation", "skillUri": "id_skill",
                 "relationType": "relation_type"}, inplace=True)
    occupation.dropna(inplace=True)
    # ---------------------------------------------------------------------
    skills = skills[["conceptUri", "preferredLabel", "skillType", "reuseLevel", "altLabels"]].astype(
        dtype={"conceptUri": "string", "preferredLabel": "string", "skillType": "string",
               "altLabels": "string", "reuseLevel": "string"})

    # remove the link (to simply the notation)
    skills["conceptUri"] = skills["conceptUri"].str.replace(SKILL_PATH, "")
    skills.rename(
        columns={"conceptUri": "id_skill", "preferredLabel": "Skill",
                 "skillType": "type", "reuseLevel": "sector", "altLabels": "synonyms"}, inplace=True)

    skills["synonyms"] = skills["synonyms"].str.split("\n")
    skills["synonyms"] = skills["synonyms"].fillna("[]")

    skills_synonyms = []
    for row in skills.itertuples():
        for syn in row[5]:
            skills_synonyms.append(dict(label=syn, id_skill=row[1]))
    skills_synonyms = pd.DataFrame(skills_synonyms)
    skills.dropna(inplace=True)
    # ---------------------------------------------------------------------
    # remove "synonyms" BEFORE store occupations and skills
    occupation.drop("synonyms", axis=1, inplace=True)
    skills.drop("synonyms", axis=1, inplace=True)

    occupation.to_csv(output_dir + "/occupations.csv", quoting=csv.QUOTE_ALL, index=False)
    occupation_synonyms.to_csv(output_dir + "/occupation_synonyms.csv", quoting=csv.QUOTE_ALL, index=False)

    skills.to_csv(output_dir + "/skills.csv", quoting=csv.QUOTE_ALL, index=False)
    skills_synonyms.to_csv(output_dir + "/skills_synonyms.csv", quoting=csv.QUOTE_ALL, index=False)

    # remove "code" and "type" BEFORE merge
    occupation.drop(["group"], axis=1, inplace=True)
    skills.drop(["type", "sector"], axis=1, inplace=True)

    job_library = occupation.merge(occ2skills.merge(skills, on="id_skill"), on="id_occupation")
    job_library = job_library[["id_occupation", "relation_type", "id_skill"]]

    job_library.to_csv(output_dir + "/job2skills.csv", index=False)

    print("Completed")


prepare_datasets(raw_sources="../raw_sources", output_dir="../sources/")
# build_cities_distance(raw_sources="../raw_sources", output_dir="../sources/", sub_sample=0.15)
