import json
import os
import random
from functools import reduce
from io import StringIO
from typing import Iterable

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv, merge, read_json
from tqdm import tqdm

from Class_utils.parameters import RelationNode, TypeNode


class JobGraph:
    OCCUPATION_GROUP_THRESHOLD = 4  # A constant for occupation (ISCO)group threshold

    def __init__(self, sources: dict, force_build: bool = False,
                 cache_path: str = None):
        """
        Occupation/Skill/Knowledge Graph
        :param sources: dictionary of paths used to load the resources
        :param force_build: if true, will be built the graph each time
        :param cache_path: path to save the cache of a graph
        """

        self.cache_path = cache_path  # string path of graph cache (json)

        # -------- Load resources --------
        # if there is the cache file then the resources will be loaded from it
        if os.path.exists(f"{self.cache_path}/graph_cache.json") and not force_build:

            print("Cache found loading...", end="")
            with open(f"{self.cache_path}/graph_cache.json", 'r') as file:
                json_data = json.load(file)

            self.occupation = read_json(StringIO(json_data[0]))  # all occupation
            self.skills = read_json(StringIO(json_data[1]))  # all skill
            self.occ2skills = read_json(StringIO(json_data[2]))  # relation "many to many"
            self.graph = nx.node_link_graph(json_data[3])  # graph
            print("done")

        else:
            # if the cache file is not found, then we load all the resources and pre-process them
            print("Cache not found, building th graph...")
            self.occupation = read_csv(sources["occupation_path"], index_col=0)  # all occupation
            self.skills = read_csv(sources["skills_path"], index_col=0)  # all skill
            self.occ2skills = read_csv(sources["job2skills_path"])  # relation "many to many"
            self.build_graph()  # pre-process the resources and build the graph
        # -------- Load resources --------

        # -------- name to id dictionary --------
        # We build a dictionary that maps the "name" (which can be "competence/knowledge" or "occupation" names)
        # into a unique (ESCO) uri (called id)
        self.name2id = {}
        self.name2id.update(dict((tuple_[1], tuple_[0]) for tuple_ in self.skills.itertuples()))
        self.name2id.update(dict((tuple_[1], tuple_[0]) for tuple_ in self.occupation.itertuples()))
        # -------- name to id dictionary --------

        # -------- Synonyms dictionary --------
        # After some experiment we concluded that the most efficient way to handle with synonyms
        # is to create two types of dictionaries:

        # "label2id_skills": dictionary that maps all skill labels (including all synonyms) into a standard uri (called
        # skill id). There are some "situations" in which we have two distinct skills that have the same synonym.
        # When we want to "standardize" a synonym, we encounter two "uri", with "attention mechanism" (see below
        # method) we select the appropriate uri (id_skill).

        # "id_skill2labels": dictionary that maps the (unique id_skill/uri) into a list of synonyms
        synonyms = read_csv(sources["skill_synonyms_path"])
        self.sys_label2id = synonyms.groupby('label')['id_skill'].apply(list).to_dict()
        self.sys_id2labels = synonyms.groupby('id_skill')['label'].apply(list).to_dict()
        # -------- Synonyms dictionary --------

    def build_graph(self):

        # ----- Data cleaning -----
        print("Data cleaning...", end="")
        # we truncate the ISCO group with max OCCUPATION_GROUP_THRESHOLD digits
        self.occupation["group"] = self.occupation["group"].str[:self.OCCUPATION_GROUP_THRESHOLD]

        # First, we retrieve all occupations that have essential competence of knowledge minus than 1
        t = self.occ2skills.merge(self.skills, on="id_skill")
        counts = t[t['relation_type'] == 'essential'].groupby(['id_occupation', 'type']).size().unstack(fill_value=0)
        filtered_occupations = counts[(counts['skill/competence'] <= 1) | (counts['knowledge'] <= 1)]
        # then we remove them
        self.occupation.drop(filtered_occupations.index, inplace=True)
        self.occ2skills = self.occ2skills[~self.occ2skills['id_occupation'].isin(filtered_occupations.index)]

        # Second, we remove all skills that are not used (Those skills that don't appear in occupations-skills relation)
        unused_skills = merge(self.skills, self.occ2skills, on='id_skill', how='left', indicator=True)
        self.skills.drop(unused_skills[unused_skills['_merge'] == 'left_only']["id_skill"])

        print("done")
        # ----- Data cleaning -----

        # ----- Populate the graph -----
        # Add occupation nodes to the graph
        progress_bar = tqdm(self.occupation.itertuples(), total=len(self.occupation), desc="Loading occupations")
        occupation_nodes = [
            (occ[0], {"type": "occupation", "label": occ[1], "isco_group": occ[2]})
            for occ in progress_bar
        ]

        # Add skill nodes to the graph
        progress_bar = tqdm(self.skills.itertuples(), total=len(self.skills), desc="Loading skills")
        skill_nodes = [
            (skill[0], {"type": skill[2], "label": skill[1], "sector": skill[3]})
            for skill in progress_bar
        ]

        # Add edges for occ2skills relations
        progress_bar = tqdm(self.occ2skills.itertuples(), total=len(self.occ2skills), desc="Add relations")
        edges_occ2skills = [
            (row[1], row[3], {"relation": row[2]})
            for row in progress_bar
        ]

        self.graph = nx.Graph()
        self.graph.add_nodes_from(occupation_nodes + skill_nodes)
        self.graph.add_edges_from(edges_occ2skills)
        # ----- Populate the graph -----

        self.save_to_cache()

    def save_to_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        occupation = self.occupation.to_json()
        skills = self.skills.to_json()
        occ2skills = self.occ2skills.to_json()
        graph = nx.node_link_data(self.graph)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(os.path.join(self.cache_path, "graph_cache.json"), 'w') as file:
            json.dump([occupation, skills, occ2skills, graph], file)

    def return_neighbors(self, id_node: str, relation: RelationNode, type_node: TypeNode,
                         exclude: list[str] = None, convert_ids: bool = False) -> list[str]:
        """
        Return a list of nodes that respect the requirement in the parameters
        :param id_node: Id-occupation or skill/knowledge
        :param relation: "essential", "optional" or "same_group" (only for occupation)
        :param type_node: "occupation", "skill" or "knowledge"
        :param exclude: list of id node to exclude
        :param convert_ids: if true coverts ids into name
        :return: list of nodes
        """
        if exclude is None:
            exclude = []

        def filter_conditions(n):
            return (
                    self.graph.edges[id_node, n]["relation"] == relation.value and
                    self.graph.nodes[n]["type"] == type_node.value and
                    n not in exclude
            )

        filtered_neighbors = [n for n in self.graph.neighbors(id_node) if filter_conditions(n)]

        if convert_ids:
            filtered_neighbors = [self.graph.nodes[n]["label"] for n in filtered_neighbors]

        return filtered_neighbors

    def sample_skills(self, id_occ: str, relation: RelationNode, type_node: TypeNode, exact_number: int = None,
                      min_: int = 2, max_: int = 4, convert_ids: bool = False, exclude: list[str] = None) -> list[str]:
        """
        Sample skill for a given occupation
        :param id_occ: id occupation
        :param relation: "essential", "optional"
        :param type_node: "occupation", "skill" or "knowledge"
        :param exact_number: if not None, the method provide an exact_number of skills
        :param min_: Min number of skills (used if exact_number is None)
        :param max_: Max number of skills
        :param exclude: lists of nodes to exclude
        :param convert_ids: if true coverts ids into name

        :return: A list of len "max_" with an "n" number of skill "min_" <= "n" <= "max_"
        """
        if exact_number is None:
            exact_number = 0 if min_ > max_ or max_ == 0 else random.randint(min_, max_)

        if exact_number == 0:
            sampled = []
        else:
            if exclude is not None:
                exclude = [self.name2id[e] for e in exclude]

            list_ = self.return_neighbors(id_occ, relation, type_node, exclude, convert_ids)
            exact_number = min(exact_number, len(list_))
            sampled = random.sample(list_, exact_number)

        if exact_number < max_:
            sampled.extend(["-" for _ in range(exact_number, max_)])

        return sampled

    def sample_occupation(self) -> tuple[str, str, str]:
        """
        Sample an occupation
        :return: id_occupation, label, isco group
        """
        id_occ = self.occupation.sample().index[0]
        return id_occ, self.graph.nodes[id_occ]["label"], self.graph.nodes[id_occ]["isco_group"]

    def get_job_with_skill(self, competences: list[str] = None, knowledge: list[str] = None) -> list[str]:
        """
        Give a list of competence & knowledge, return a list of id-occupation that has (at least) these skills
        :return:
        """
        if competences is None:
            competences = []
        if knowledge is None:
            knowledge = []

        skills = competences + knowledge
        if len(skills) < 1:
            return []

        nodes = [set(self.graph.neighbors(self.name2id[skill])) for skill in skills]
        if nodes:
            similar_jobs = reduce(set.intersection, nodes)
            return list(similar_jobs)
        else:
            return []

    def get_single_component(self) -> list[str]:
        """
        Returns a list of single component in the graph
        """
        return [node for component in nx.connected_components(self.graph)
                if len(component) == 1 for node in component]

    def node_similarity(self, nodesA: list | set, nodesB: list | set, ids: bool = False) -> list[float]:
        """
        Give a set/list of nodes "A" and a set/list of nodes B, returns a list of jaccard coefficients.
        The length of the list is equal to the length of nodesA. The one jaccard coefficient represents the
         sum of jaccard similarity between a node A and all nodes in B.
        :param nodesA: set/list of nodes
        :param nodesB: set/list of nodes
        :param ids: True is the nodes are already (skill_id)
        :return: a list of jaccard coefficients
        """
        if len(nodesA) == 0 or len(nodesB) == 0:
            return [0]

        if not ids:
            nodesA = [self.name2id[node] for node in nodesA]
            nodesB = [self.name2id[node] for node in nodesB]

        list_ = [(nodeA, nodesB) for nodeA in nodesA]  # for one node in A => all nodes in B

        similarity = []
        for nodeA, nodesB in list_:  # nodeA (single node) , nodesB (all nodes)
            tuples = [(nodeA, nodeB) for nodeB in nodesB]
            sim_ = sum(coeff for _, _, coeff in nx.jaccard_coefficient(self.graph, tuples))
            similarity.append(sim_)
        return similarity

    def jaccard_coefficient(self, nodeA: str, nodeB: str, ids: bool = False):
        if not ids:
            nodeA, nodeB = self.name2id[nodeA], self.name2id[nodeB]
        return nx.jaccard_coefficient(self.graph, [(nodeA, nodeB)])

    def return_all_neighbors_info(self, id_node: str) -> list:

        def info(n):
            return n, self.graph.nodes[n]["label"]

        filtered_neighbors = [info(n) for n in self.graph.neighbors(id_node)]
        return filtered_neighbors

    def show_subgraph(self, id_occ: str, max_nodes: int = 0):

        color_node_map = {"occupation": "Red", "knowledge": "#95a6df", "skill/competence": "#ffa500"}
        color_edge_map = {"essential": 1, "optional": 1, "transversal": 1}

        nodes = self.return_all_neighbors_info(id_occ)
        random.shuffle(nodes)
        if max_nodes > 0:
            nodes = nodes[:max_nodes]
        nodes.append((id_occ, self.graph.nodes[id_occ]["label"]))

        subgraph = self.graph.subgraph([node[0] for node in nodes])
        labels = {node[0]: node[1] for node in nodes}

        node_color = [color_node_map[self.graph.nodes[node]["type"]] for node in subgraph]
        edge_colors = [color_edge_map[self.graph.edges[edge]['relation']] for edge in subgraph.edges]
        node_size = [1200 if node == id_occ else 300 for node in subgraph]

        pos = nx.shell_layout(subgraph)
        plt.figure(figsize=(20, 8))
        nx.draw(subgraph, pos, with_labels=True, labels=labels, width=edge_colors,
                node_color=node_color, node_size=node_size, font_size=20, font_family='Arial', edge_color='black')
        plt.show()

    def substitute_skills(self, mask: list[bool], skills: Iterable[str], ids: bool = False) -> list[str]:
        """
        Given an iterable skills and a mask of booleans with equal length, returns a list of synonyms.
        """

        def map_skill(m: bool, s: str) -> str:
            if s == "-" or not m:
                return s
            if not ids:
                s = self.name2id[s]

            # given an uri, we return a list of synonyms
            synonyms_skills = self.sys_id2labels[s]
            # if there exist, we sample one synonym
            return random.choice(synonyms_skills) if len(synonyms_skills) > 0 else s

        return [map_skill(m, s) for m, s in zip(mask, skills)]

    def skill_standardize(self, skills: Iterable[str], to_ids: bool = True) -> set:
        """
        Given a list of skills, we return a correspondent standard uri. If there is "ambiguous" situation
        in which one synonym can be associated with multiple standard uri. We apply the "attention" mechanism
        to understand which is the appropriate uri.
        """

        # map the skill into standard uri, if there is "ambiguation", the dictionary returns a list of possible uri
        skills = [self.sys_label2id[skill] for skill in skills]

        # "unique_uri" represent a list of non-ambiguous synonyms (that can be associated only to one "standard" label)
        unique_uri = [skill[0] for skill in skills if len(skill) == 1]
        # "ambiguous_uri" is a list of "ambiguous" uri, list[list[str]]
        ambiguous_uri = [skill for skill in skills if len(skill) > 1]

        # ---- attention mechanism base on the contex ----
        # for all ambiguous synonyms we have a list of possible id_skills. We perform the "node similarity" between
        # the id_skill and the all unique_uri. The id_skill that achieves the higher result is the most appropriate one.
        de_ambiguous_uri = [ambiguous[np.argmax(self.node_similarity(ambiguous, unique_uri, True))]
                            for ambiguous in ambiguous_uri]
        # ---- attention mechanism base on the contex ----

        if not to_ids:
            unique_uri = [self.graph.nodes[i]["label"] for i in unique_uri]
            de_ambiguous_uri = [self.graph.nodes[i]["label"] for i in de_ambiguous_uri]

        return set(unique_uri + de_ambiguous_uri)
