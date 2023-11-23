import json
import os
import random
from functools import reduce
from io import StringIO

import networkx as nx
from matplotlib import pyplot as plt
from pandas import read_csv, merge, read_json
from tqdm import tqdm

from Class_utils.parameters import RelationNode, TypeNode


class JobGraph:
    OCCUPATION_GROUP_THRESHOLD = 4  # A constant for occupation group threshold
    occ_weight = {0: 42, 1: 19, 2: 8, 3: 3, 4: 1}

    def __init__(self, sources: dict, force_build: bool = False,
                 cache_path: str = None):
        """
        Occupation/Skill/Knowledge Graph
        """
        self.cache_path = cache_path
        self.graph = None
        # -------- Load resources --------
        if os.path.exists(f"{self.cache_path}/graph_cache.json") and not force_build:

            print("Cache found loading...", end="")
            with open(f"{self.cache_path}/graph_cache.json", 'r') as file:
                json_data = json.load(file)

            self.occupation = read_json(StringIO(json_data[0]))  # all occupation
            self.skills = read_json(StringIO(json_data[1]))  # all skill
            self.occ2skills = read_json(StringIO(json_data[2]))  # relation "many to many"
            self.graph = nx.node_link_graph(json_data[3])
            print("done")

        else:
            print("Cache not found, building th graph...")

            self.occupation = read_csv(sources["occupation_path"], index_col=0)  # all occupation
            self.skills = read_csv(sources["skills_path"], index_col=0)  # all skill
            self.occ2skills = read_csv(sources["job2skills_path"])  # relation "many to many"
            self.build_graph()
        # -------- Load resources --------

        self.name2id = {}
        self.name2id.update(dict((tuple_[1], tuple_[0]) for tuple_ in self.skills.itertuples()))
        self.name2id.update(dict((tuple_[1], tuple_[0]) for tuple_ in self.occupation.itertuples()))

    def build_graph(self):

        # ----- Data cleaning -----
        print("Data cleaning...", end="")
        self.occupation["group"] = self.occupation["group"].str[:self.OCCUPATION_GROUP_THRESHOLD]

        # First, we retrieve all occupations that have essential competence of knowledge minus than 1
        t = self.occ2skills.merge(self.skills, on="id_skill")
        counts = t[t['relation_type'] == 'essential'].groupby(['id_occupation', 'type']).size().unstack(fill_value=0)
        filtered_occupations = counts[(counts['skill/competence'] <= 1) | (counts['knowledge'] <= 1)]
        # We remove them
        self.occupation.drop(filtered_occupations.index, inplace=True)
        self.occ2skills = self.occ2skills[~self.occ2skills['id_occupation'].isin(filtered_occupations.index)]
        # Then we remove all skills that are not used (Those skills that don't appear in occupations-skills relation)
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
        occupation = self.occupation.to_json()
        skills = self.skills.to_json()
        occ2skills = self.occ2skills.to_json()
        graph = nx.node_link_data(self.graph)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        with open(f"{self.cache_path}/graph_cache.json", 'w') as file:
            json.dump([occupation, skills, occ2skills, graph], file)

    def weight_group_node(self, groupA: str, groupB: str):
        lvl = 0
        while lvl <= 3:
            if groupA[lvl] != groupB[lvl]:
                return self.occ_weight[lvl]
            else:
                lvl += 1
        return self.occ_weight[lvl]

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

        def filter_condition(n):
            return (
                    self.graph.edges[id_node, n]["relation"] == relation.value and
                    self.graph.nodes[n]["type"] == type_node.value and
                    n not in exclude
            )

        filtered_neighbors = [n for n in self.graph.neighbors(id_node) if filter_condition(n)]

        if convert_ids:
            filtered_neighbors = [self.graph.nodes[n]["label"] for n in filtered_neighbors]

        return filtered_neighbors

    def sample_skills(self, id_occ: str, relation: RelationNode, type_node: TypeNode, exact_number: int = None,
                      min_: int = 2, max_: int = 4, convert_ids: bool = False, exclude: list[str] = None) -> list[str]:
        """
        Sample skill for a given occupation
        :param id_occ: id occupation
        :param relation: "essential", "optional" or "same_group" (only for occupation)
        :param type_node: "occupation", "skill" or "knowledge"
        :param exact_number:
        :param min_: Min number of skills
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

    def get_path(self, nodeA: str, nodeB: str, ids: bool = False, convert_ids: bool = False) -> list[tuple]:
        if not ids:
            nodeA, nodeB = self.name2id[nodeA], self.name2id[nodeB]

        nodes = nx.shortest_path(self.graph, source=nodeA, target=nodeB)

        if convert_ids:
            nodes = [(self.graph.nodes[node]["label"], self.graph.nodes[node]["type"]) for node in nodes]
        else:
            nodes = [(node, self.graph.nodes[node]["type"]) for node in nodes]

        return nodes

    def remove_single_component(self) -> list[str]:

        return [node for component in nx.connected_components(self.graph)
                if len(component) == 1 for node in component]

    def node_similarity(self, nodesA: list[str], nodesB: list[str], ids: bool = False) -> float:

        if len(nodesA) == 0 or len(nodesB) == 0:
            return 0

        if not ids:
            nodesA = [self.name2id[node] for node in nodesA]
            nodesB = [self.name2id[node] for node in nodesB]

        list_ = [(nodeA, nodeB) for nodeA in nodesA for nodeB in nodesB]

        avg_similarity = 0
        for _, _, coeff in nx.jaccard_coefficient(self.graph, list_):
            avg_similarity += coeff
        return avg_similarity / len(list_)

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
        color_node_map = {"occupation": "#3B8183", "knowledge": "#95a6df", "skill/competence": "#ffa500"}
        color_edge_map = {"essential": 2, "optional": 1, "transversal": 1}

        nodes = self.return_all_neighbors_info(id_occ)
        random.shuffle(nodes)
        if max_nodes > 0:
            nodes = nodes[:max_nodes]
        nodes.append((id_occ, self.graph.nodes[id_occ]["label"]))

        subgraph = self.graph.subgraph([node[0] for node in nodes])
        labels = {node[0]: node[1] for node in nodes}

        node_color = [color_node_map[self.graph.nodes[node]["type"]] for node in subgraph]
        edge_colors = [color_edge_map[self.graph.edges[edge]['relation']] for edge in subgraph.edges]

        node_size = [600 if node == id_occ else 300 for node in subgraph]

        pos = nx.spring_layout(subgraph)
        plt.figure(figsize=(15, 10))
        nx.draw(subgraph, pos, with_labels=True, labels=labels, width=edge_colors,
                node_color=node_color, node_size=node_size)
        plt.show()
