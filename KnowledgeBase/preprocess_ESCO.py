import pandas as pd

# Constant
OCCUPATION_PATH = "http://data.europa.eu/esco/occupation/"
SKILL_PATH = "http://data.europa.eu/esco/skill/"
OCCUPATION_GROUP_THRESHOLD = 4  # A constant for occupation (ISCO)group threshold

# Input files
OCCUPATIONS_FILE = './../raw_sources/occupations_en.csv'
SKILLS_FILE = './../raw_sources/skills_en.csv'
ESCO_LANGUAGE_FILE = './../raw_sources/languageSkillsCollection_en.csv'
OCC2SKILLS_FILE = './../raw_sources/occupationSkillRelations_en.csv'

# Output files
SKILLS_TO_SYS = "./../sources/skillUri2sys.json"
SYS_TO_SKILLS = "./../sources/sys2skillUri.json"
OCCUPATION_DESCRIPTION = "./../sources/occupation_description.json"
SKILLS_DESCRIPTION = "./../sources/skills_description.json"
OCC2SKILLS = "./../sources/occ2skills.json"
LANGUAGE_ESCO = "./../sources/language_esco.json"
OCCUPATION = "./../sources/occupations.json"
SKILLS = "./../sources/skills.json"


def strip_list(lst):
    # Define a function to strip each item in a list
    return [x.strip() for x in lst]


# ====================== occupation ======================
occupation = pd.read_csv(OCCUPATIONS_FILE)
occupation = occupation[["conceptUri", "code", "preferredLabel", "altLabels", "description"]].astype(
    dtype={
        "conceptUri": "string",
        "preferredLabel": "string",
        "code": "string",
        "altLabels": "string",
        "description": "string"})

occupation["conceptUri"].replace(OCCUPATION_PATH, "", regex=True, inplace=True)
occupation.rename(
    columns={"conceptUri": "occUri", "preferredLabel": "occupation", "altLabels": "synonyms", "code": "iscoGroup"},
    inplace=True)
occupation.dropna(subset=["occupation"], inplace=True)
occupation["occupation"] = occupation["occupation"].str.strip()
occupation["synonyms"] = occupation["synonyms"].str.split("\n")
occupation["synonyms"] = occupation["synonyms"].apply(lambda x: strip_list(x) if isinstance(x, list) else x)
occupation["synonyms"].fillna(value="", inplace=True)
occupation["iscoGroup"] = "G" + occupation["iscoGroup"].str[:OCCUPATION_GROUP_THRESHOLD]
# ====================== occupation ======================

# ====================== skills ======================
skills = pd.read_csv(SKILLS_FILE)
skills = skills[["conceptUri", "skillType", "reuseLevel", "preferredLabel", "altLabels", "description"]].astype(
    dtype={"conceptUri": "string", "preferredLabel": "string", "skillType": "string",
           "altLabels": "string", "reuseLevel": "string", "description": "string"})

skills["conceptUri"].replace(SKILL_PATH, "", regex=True, inplace=True)
skills.rename(
    columns={"conceptUri": "skillUri", "preferredLabel": "skill", "reuseLevel": "sector", "altLabels": "synonyms"},
    inplace=True)
skills.dropna(subset=["skill", "skillType", "sector"], inplace=True)
skills["skill"] = skills["skill"].str.strip()
skills["synonyms"] = skills["synonyms"].str.split("\n")
skills["synonyms"] = skills["synonyms"].apply(lambda x: strip_list(x) if isinstance(x, list) else x)
skills["synonyms"].fillna(value="", inplace=True)
# ====================== skills ======================

# ====================== Occ2Skill ======================
occ2skills = pd.read_csv(OCC2SKILLS_FILE)
occ2skills = occ2skills.astype(dtype={"occupationUri": "string", "relationType": "string", "skillUri": "string"})
occ2skills.rename(columns={"occupationUri": "occUri"}, inplace=True)
occ2skills["occUri"].replace(OCCUPATION_PATH, "", regex=True, inplace=True)
occ2skills["skillUri"].replace(SKILL_PATH, "", regex=True, inplace=True)
occ2skills.drop(["skillType"], inplace=True, axis=1)
print("Occ2skills relations", len(occ2skills))
# ====================== Occ2Skill ======================

# ====================== languages ======================
languages = pd.read_csv(ESCO_LANGUAGE_FILE)
languages = languages[["conceptUri", "skillType", "preferredLabel", "altLabels"]].copy()
languages["conceptUri"].replace(SKILL_PATH, "", regex=True, inplace=True)
languages.rename(columns={"conceptUri": "langUri", "preferredLabel": "skill", "altLabels": "synonyms"}, inplace=True)
languages.dropna(subset=["skill"], inplace=True)
languages["skill"] = languages["skill"].str.strip()
languages["synonyms"] = languages["synonyms"].str.split("|").apply(
    lambda x: strip_list(x) if isinstance(x, list) else x)
languages["synonyms"].fillna(value="", inplace=True)
# ====================== languages ======================


# Remove languages from skills
to_remove = skills[skills["skillUri"].isin(languages["langUri"])].index
skills.drop(to_remove, inplace=True)
print("Number of skills removed (language skills)", len(to_remove))

# Remove relations that are present in "occ2skills" but not in occupation or skills
temp = pd.merge(occ2skills, occupation[["occUri", "occupation"]], on='occUri', how="left")
not_present = temp[temp.isna().any(axis=1)]["occUri"].unique().tolist()
to_remove = occ2skills[occ2skills["occUri"].isin(not_present)].index
occ2skills.drop(to_remove, inplace=True, axis=0)
print("Number of elements present in Occ2Skills but not in Occupations", len(not_present), "removed", len(to_remove))

temp = pd.merge(occ2skills, skills[["skillUri", "skill"]], on='skillUri', how="left")
not_present = temp[temp.isna().any(axis=1)]["skillUri"].unique().tolist()
to_remove = occ2skills[occ2skills["skillUri"].isin(not_present)].index
occ2skills.drop(to_remove, inplace=True, axis=0)
print("Number of elements present in Occ2Skills but not in Skills", len(not_present), "removed", len(to_remove))

# # Mark all occupations that have essential competence or knowledge minus than 2
temp = occ2skills.query("relationType=='essential'").merge(skills, on="skillUri")[
    ["occUri", "relationType", "skillUri", "skillType"]]
ess_competences, ess_knowledge = temp.query("skillType == 'skill/competence'"), temp.query("skillType == 'knowledge'")

occUri_lack1 = ess_competences.groupby("occUri").count().merge(occupation["occUri"],
                                                               left_index=True, right_on="occUri",
                                                               how="right").fillna(0).query("skillType < 2")["occUri"]
occUri_lack2 = ess_knowledge.groupby("occUri").count().merge(occupation["occUri"],
                                                             left_index=True, right_on="occUri",
                                                             how="right").fillna(0).query("skillType < 2")["occUri"]
occUri_lack = set(occUri_lack1.to_list() + occUri_lack2.to_list())

occupation["sample"] = ~occupation["occUri"].isin(occUri_lack)
print("Num of occupation that cannot be sampled in generator because has less than essential 2 competence/knowledge",
      len(occUri_lack))

# save dataframe
uri2sys = skills[skills["synonyms"] != ""][["skillUri", "synonyms"]]
uri2sys.to_json(SKILLS_TO_SYS, indent=2, orient="records")

sys2uri = skills[skills["synonyms"] != ""][["skillUri", "synonyms"]]
map1 = sys2uri.explode('synonyms').groupby("synonyms")["skillUri"].apply(list)
map2 = skills[["skill", "skillUri"]].groupby("skill")["skillUri"].apply(list)
sys2uri = pd.concat([map1, map2], axis=0)
sys2uri = sys2uri[~sys2uri.index.duplicated(keep='first')]
sys2uri.to_json(SYS_TO_SKILLS, indent=2, orient="index")

occupation[["occUri", "description"]].to_json(OCCUPATION_DESCRIPTION, indent=2, orient="records")
skills[["skillUri", "description"]].to_json(SKILLS_DESCRIPTION, indent=2, orient="records")
occ2skills.to_json(OCC2SKILLS, indent=2, orient="records")
languages.to_json(LANGUAGE_ESCO, indent=2, orient="records")

occupation.drop(["description", "synonyms"], axis=1, inplace=True)
skills.drop(["description", "synonyms"], axis=1, inplace=True)

occupation.to_json(OCCUPATION, indent=2, orient="records")
skills.to_json(SKILLS, indent=2, orient="records")

print("Done")
