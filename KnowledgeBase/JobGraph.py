import json
import os
import random
from functools import reduce
from typing import Iterable

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_json
from tqdm import tqdm

from Class_utils.parameters import RelationNode, TypeNode


class JobGraph:

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
        self.occupation = read_json(sources["occupation_path"]).set_index("occUri")  # all occupation
        self.skills = read_json(sources["skills_path"]).set_index("skillUri")  # all skill
        self.occ2skills = read_json(sources["occ2skills_path"])  # relation "many to many"

        # After some experiment, we concluded that the most efficient way to handle with synonyms
        # is to create two types of dictionaries:
        # "label2uri_skills": dictionary that maps all skill labels (including all synonyms) into a standard uri (called
        # skill id). There are some "situations" in which we have two distinct skills that have the same synonym.
        # When we want to "standardize" a synonym, we encounter two "uri", with "contex mechanism" (see below
        # method) we select the appropriate uri (uri_skill).
        self.skillUri2sys = read_json(sources["skillUri2sys_path"]).set_index("skillUri").to_dict()["synonyms"]
        self.sys2skillUri = read_json(sources["sys2skillUri_path"], typ="series").to_dict()

        # if there is the cache file, then the resources will be loaded from it
        if os.path.exists(f"{self.cache_path}/graph_cache.json") and not force_build:

            print("Cache found loading...", end="")
            with open(f"{self.cache_path}/graph_cache.json", 'r') as file:
                json_data = json.load(file)
            self.graph = nx.node_link_graph(json_data)  # graph
            print("done")

        else:
            # if the cache file is not found, then we load all the resources and pre-process them
            print("Cache not found, building th graph...")
            self.graph = self.build_graph()  # pre-process the resources and build the graph
        # -------- Load resources --------

        # -------- name to id dictionary --------
        # We build a dictionary that maps the "name" (which can be "competence/knowledge" or "occupation" names)
        # into a unique (ESCO) uri (called id)
        self.name2uri = {}
        self.name2uri.update(dict((tuple_[3], tuple_[0]) for tuple_ in self.skills.itertuples()))
        self.name2uri.update(dict((tuple_[2], tuple_[0]) for tuple_ in self.occupation.itertuples()))
        # -------- name to id dictionary --------

    def build_graph(self):

        # ----- Populate the graph -----
        # Add occupation nodes to the graph
        progress_bar = tqdm(self.occupation.itertuples(), total=len(self.occupation), desc="Loading occupations")
        occupation_nodes = [
            (occ[0], {"type": "occupation", "isco_group": occ[1], "label": occ[2], "sample": occ[3]})
            for occ in progress_bar
        ]

        # Add skill nodes to the graph
        progress_bar = tqdm(self.skills.itertuples(), total=len(self.skills), desc="Loading skills")
        skill_nodes = [
            (skill[0], {"type": skill[1], "sector": skill[2], "label": skill[3]})
            for skill in progress_bar
        ]

        # Add edges for occ2skills relations
        progress_bar = tqdm(self.occ2skills.itertuples(), total=len(self.occ2skills), desc="Add relations")
        edges_occ2skills = [
            (row[1], row[3], {"relationType": row[2]})
            for row in progress_bar
        ]

        graph = nx.Graph()
        graph.add_nodes_from(occupation_nodes + skill_nodes)
        graph.add_edges_from(edges_occ2skills)
        # ----- Populate the graph -----

        self.save_to_cache(graph)
        return graph

    def save_to_cache(self, graph):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        graph = nx.node_link_data(graph)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(os.path.join(self.cache_path, "graph_cache.json"), 'w') as file:
            json.dump(graph, file)

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
                    self.graph.edges[id_node, n]["relationType"] == relation.value and
                    self.graph.nodes[n]["type"] == type_node.value and
                    n not in exclude
            )

        filtered_neighbors = [n for n in self.graph.neighbors(id_node) if filter_conditions(n)]

        if convert_ids:
            filtered_neighbors = [self.graph.nodes[n]["label"] for n in filtered_neighbors]

        return filtered_neighbors

    def sample_skills(self, uri_occ: str, relation: RelationNode, type_node: TypeNode, num: int,
                      convert_ids: bool = False, exclude: list[str] = None) -> list[str]:
        """
        Sample skill for a given occupation
        :param uri_occ: id occupation
        :param relation: "essential", "optional"
        :param type_node: "occupation", "skill" or "knowledge"
        :param num: number of skills (at max) to sample
        :param exclude: lists of nodes to exclude
        :param convert_ids: if true coverts ids into name

        :return: A list of "n" number of skills "min_" <= "n" <= "max_"
        """
        if num == 0:
            sampled = []
        else:
            if exclude is not None:
                exclude = [self.name2uri[e] for e in exclude]

            list_ = self.return_neighbors(uri_occ, relation, type_node, exclude, convert_ids)
            exact_number = min(num, len(list_))
            sampled = random.sample(list_, exact_number)

        return sampled

    def sample_occupation(self) -> tuple[str, str, str]:
        """
        Sample an occupation
        :return: id_occupation, label, isco group
        """
        uri_occ = self.occupation.query("sample==True").sample().index[0]
        return uri_occ, self.graph.nodes[uri_occ]["label"], str(self.graph.nodes[uri_occ]["isco_group"])

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

        nodes = [set(self.graph.neighbors(self.name2uri[skill])) for skill in skills]
        if nodes:
            similar_jobs = reduce(set.intersection, nodes)
            sample_occ = self.occupation.query("sample==True").index
            similar_jobs = [j for j in similar_jobs if j in sample_occ]
            return similar_jobs
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
            nodesA = [self.name2uri[node] for node in nodesA]
            nodesB = [self.name2uri[node] for node in nodesB]

        list_ = [(nodeA, nodesB) for nodeA in nodesA]  # for one node in A => all nodes in B

        similarity = []
        for nodeA, nodesB in list_:  # nodeA (single node) , nodesB (all nodes)
            tuples = [(nodeA, nodeB) for nodeB in nodesB]
            sim_ = sum(coeff for _, _, coeff in nx.jaccard_coefficient(self.graph, tuples)) / len(nodesB)
            similarity.append(sim_)
        return similarity

    def jaccard_coefficient(self, nodeA: str, nodeB: str, ids: bool = False):
        if not ids:
            nodeA, nodeB = self.name2uri[nodeA], self.name2uri[nodeB]
        return nx.jaccard_coefficient(self.graph, [(nodeA, nodeB)])

    def return_all_neighbors_info(self, id_node: str) -> list:

        def info(n):
            return n, self.graph.nodes[n]["label"]

        filtered_neighbors = [info(n) for n in self.graph.neighbors(id_node)]
        return filtered_neighbors

    def substitute_skills(self, mask: list[bool], skills: Iterable[str]) -> list[str]:
        """
        Given an iterable skills and a mask of booleans with equal length, returns a list of synonyms.
        """

        def map_skill(sub_mask: bool, label: str) -> str:
            if not sub_mask:
                return label
            uri = self.name2uri[label]
            # Given an uri, we return a list of synonyms. If there exist, we sample one synonym
            synonyms_skills = self.skillUri2sys[uri] if uri in self.skillUri2sys else [label]
            return random.choice(synonyms_skills)

        return [map_skill(m, s) for m, s in zip(mask, skills)]

    def skill_standardize(self, skills: Iterable[str]) -> tuple[list[str], list[list[str]]]:

        # map the skill into standard uri, if there is "ambiguation", the dictionary returns a list of possible uri
        skills = [self.sys2skillUri[skill] for skill in skills]

        # "unique_uri" represent a list of non-ambiguous synonyms (that can be associated only to one "standard" label)
        unique_uri = [skill[0] for skill in skills if len(skill) == 1]
        # "ambiguous_uri" is a list of "ambiguous" uri, list[list[str]]
        ambiguous_uri = [skill for skill in skills if len(skill) > 1]

        return unique_uri, ambiguous_uri

    def solve_ambiguous(self, ambiguous_uri: Iterable[list[str]], contex_uri: list[str], to_ids: bool = True):

        # ---- contex mechanism base on the contex ----
        # for all ambiguous synonyms we have a list of possible uri_skills.
        # We perform the "node similarity" between the uri_skill and the all unique_uri.
        # The uri_skill that achieves the higher result is the most appropriate one.
        de_ambiguous_uri = [ambiguous[np.argmax(self.node_similarity(ambiguous, contex_uri, True))]
                            for ambiguous in ambiguous_uri]
        # ---- contex mechanism base on the contex ----

        if not to_ids:
            de_ambiguous_uri = [self.graph.nodes[i]["label"] for i in de_ambiguous_uri]
        return de_ambiguous_uri

    def map_names2uri(self, names: Iterable[str], set_: bool = False) -> list[str] | set[str]:
        output = [self.name2uri[skill] for skill in names if skill != "-"]
        if set_:
            output = set(output)
        return output

    def show_subgraph(self, uri_occ: str, max_nodes: int = 0):

        color_node_map = {"occupation": "Red", "knowledge": "#95a6df", "skill/competence": "#ffa500"}
        color_edge_map = {"essential": 1, "optional": 1, "transversal": 1}

        nodes = self.return_all_neighbors_info(uri_occ)
        random.shuffle(nodes)
        if max_nodes > 0:
            nodes = nodes[:max_nodes]
        nodes.append((uri_occ, self.graph.nodes[uri_occ]["label"]))

        subgraph = self.graph.subgraph([node[0] for node in nodes])
        labels = {node[0]: node[1] for node in nodes}

        node_color = [color_node_map[self.graph.nodes[node]["type"]] for node in subgraph]
        edge_colors = [color_edge_map[self.graph.edges[edge]['relation']] for edge in subgraph.edges]
        node_size = [1200 if node == uri_occ else 300 for node in subgraph]

        pos = nx.spring_layout(subgraph, seed=42, k=0.9)
        plt.figure(figsize=(20, 8))
        nx.draw(subgraph, pos, with_labels=True, labels=labels, width=edge_colors,
                node_color=node_color, node_size=node_size, font_size=20, font_family='Arial',
                edge_color='black', alpha=0.1)
        plt.show()
