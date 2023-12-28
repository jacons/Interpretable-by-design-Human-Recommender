import csv

import pandas as pd
from geopy.distance import geodesic
from pandas import DataFrame
from tqdm import tqdm
from unidecode import unidecode


def build_cities_distance(raw_sources: str, output_dir: str, sub_sample: float = 0.1) -> DataFrame:
    tqdm.pandas()

    print("Loading source data...", end="")
    df_geo = pd.read_json(raw_sources + "/italy_geo.json")
    df_cities = pd.read_json(raw_sources + "/italy_cities.json")

    df = df_cities.merge(df_geo[["istat", "lng", "lat"]], on="istat")
    print("Completed")

    print("Data cleaning...", end="")
    # Translate "comune" and "num_residenti" in english
    df.rename(columns={"comune": "city", "num_residenti": "num_residents"}, inplace=True)

    # Drop useless columns
    df = df.drop(["istat"], axis=1).dropna().drop(df[df["city"] == ""].index)
    # Remove special character like à,è ecc...
    df["city"] = df["city"].progress_apply(lambda name: unidecode(name))

    # Reduce the dimension of the file and combine the cities
    df = df.sample(frac=sub_sample)

    df["P"] = df["num_residents"] / df["num_residents"].sum()
    df[["city", "num_residents", "P"]].to_csv(output_dir + "/all_cities.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    df.drop(["num_residents"], axis=1, inplace=True)
    print("Completed")

    print("Merging...", end="")
    df = df.merge(df, how='cross')
    # Remove the cities with the same name
    df.drop(df[df["city_x"] == df["city_y"]].index, axis=0, inplace=True)
    print("Completed")

    print("Optimizing...", end="")
    # Remove all duplicate rows Es. A->B and B->A
    df.sort_values(["city_x", "city_y"], inplace=True)
    df['temp'] = df.progress_apply(lambda row: ''.join(sorted([row['city_x'], row['city_y']])), axis=1)
    df = df.drop_duplicates(subset='temp').drop(columns=['temp'])
    print("Completed")

    print("Performing distances...", end="")
    # Calculate the distance between the cities
    df["Dist"] = df.progress_apply(lambda x: geodesic((x["lat_x"], x["lng_x"]), (x["lat_y"], x["lng_y"])).km, axis=1)
    print("Completed")

    print("Formatting...", end="")
    # Casting and sorting
    df = df[["city_x", "city_y", "Dist"]].astype({"city_x": "string", "city_y": "string", "Dist": "int"})
    df.sort_values(["city_x", "city_y", "Dist"], inplace=True)
    df.to_csv(output_dir + "/cities_distance.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print("Completed")

    return df

# build_cities_distance(raw_sources="../raw_sources", output_dir="../sources/", sub_sample=0.20)
