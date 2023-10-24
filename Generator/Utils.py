import pandas as pd
from geopy.distance import geodesic
from pandas import DataFrame
from unidecode import unidecode


def build_citizen_distance(path: str = None,
                           c_distance_path: str = None,
                           all_citizen_path: str = None,
                           sub_sample: float = 0.1) -> DataFrame:
    ITALIAN_CITIZEN = "https://raw.githubusercontent.com/MatteoHenryChinaski" + \
                      "/Comuni-Italiani-2018-Sql-Json-excel/master/italy_geo.json"

    path = ITALIAN_CITIZEN if path is None else path

    print("Downloading and Loading source data...", end="")
    df = pd.read_json(path)
    print("Completed")

    print("Data cleaning...", end="")
    # Drop useless columns
    df = df.drop(["istat"], axis=1).dropna().drop(df[df["comune"] == ""].index)
    # Remove special character like à,è ecc...
    df["comune"] = df["comune"].apply(lambda name: unidecode(name))
    # Reduce the dimension of the file and combine the citizen
    df = df.sample(frac=sub_sample)

    all_citizen_path = "all_cities.csv" if all_citizen_path is None else all_citizen_path
    df["comune"].to_csv(all_citizen_path, index=False)
    print("Completed")

    print("Merging...", end="")
    df = df.merge(df, how='cross')
    # Remove the citizen with the same name
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
    # Calculate the distance between the citizen
    df["D"] = df.apply(lambda x: geodesic((x["lat_x"], x["lng_x"]), (x["lat_y"], x["lng_y"])).km, axis=1)
    print("Completed")

    print("Formatting...", end="")
    # Casting and sorting
    df = df[["A", "B", "D"]].astype({"A": "string", "B": "string", "D": "int"})
    df.sort_values(["A", "B", "D"], inplace=True)
    output_file = "cities_distance.csv" if c_distance_path is None else c_distance_path
    df.to_csv(output_file, index=False)
    print("Completed")

    return df


build_citizen_distance(sub_sample=0.15, c_distance_path="../sources/cities_distance.csv",
                       all_citizen_path="../sources/all_cities.csv")
# %%
