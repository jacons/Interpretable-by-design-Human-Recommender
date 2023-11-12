import itertools
import random
from functools import reduce

import networkx as nx
from matplotlib import pyplot as plt
from pandas import read_csv

from Class_utils.parameters import RelationNode, TypeNode


class JobGraph:
    OCCUPATION_GROUP_THRESHOLD = 4  # A constant for occupation group threshold
    occ_weight = {0: 42, 1: 19, 2: 8, 3: 3, 4: 1}

    def __init__(self, sources: dict):
        """
        Occupation/Skill/Knowledge Graph
        """
        self.occ2skills = read_csv(sources["job2skills_path"])  # relation "many to many"
        self.occupation = read_csv(sources["occupation_path"], index_col=0)  # all occupation
        self.skills = read_csv(sources["skills_path"], index_col=0)  # all skill

        # remove an occupation that hasn't "essential skills"
        self.occupation = self.occupation[self.occupation.index != "a580e79a-b752-49c1-b033-b5ab2b34bfba"]
        self.occupation["group"] = self.occupation["group"].str[:self.OCCUPATION_GROUP_THRESHOLD]

        # Add occupation nodes to the graph
        occupation_nodes = [
            (occ[0], {"type": "occupation", "label": occ[1], "isco_group": occ[2]})
            for occ in self.occupation.itertuples()
        ]

        # Add skill nodes to the graph
        skill_nodes = [
            (skill[0], {"type": skill[2], "label": skill[1], "sector": skill[3]})
            for skill in self.skills.itertuples()
        ]

        # Add edges for occ2skills relations
        edges_occ2skills = [
            (row[1], row[3], {"relation": row[2]})
            for row in self.occ2skills.itertuples()
        ]

        # Link together the occupation with the same group
        edges_group_nodes = [
            (occ1[0], occ2[0], {"relation": "same_group", "weight": self.weight_group_node(occ1[2], occ2[2])})
            for occ1, occ2 in itertools.product(self.occupation.itertuples(), repeat=2)
            if occ1[0] != occ2[0]
        ]

        self.name2id = {}
        self.name2id.update(dict((tuple_[1], tuple_[0]) for tuple_ in self.skills.itertuples()))
        self.name2id.update(dict((tuple_[1], tuple_[0]) for tuple_ in self.occupation.itertuples()))

        self.graph = nx.Graph()
        self.graph.add_nodes_from(occupation_nodes + skill_nodes)
        self.graph.add_edges_from(edges_occ2skills + edges_group_nodes)
        self.graph.remove_nodes_from(self.remove_single_component())

    def weight_group_node(self, groupA: str, groupB: str):
        lvl = 0
        while lvl <= 3:
            if groupA[lvl] != groupB[lvl]:
                return self.occ_weight[lvl]
            else:
                lvl += 1
        return self.occ_weight[lvl]

    def return_neighbors(self, id_node: str,
                         relation: RelationNode,
                         type_node: TypeNode,
                         exclude: list[str] = None,
                         convert_ids: bool = False) -> list[str]:
        """
        Return a list of nodes that respect the requirement in the parameters
        :param id_node: Id of occupation or skill/knowledge
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

    def sample_skills(self,
                      id_occ: str,
                      relation: RelationNode,
                      type_node: TypeNode,
                      min_: int = 2, max_: int = 6,
                      convert_ids: bool = False,
                      exclude: list[str] = None):
        """
        Sample skill for a given occupation
        :param id_occ: id occupation
        :param relation: "essential", "optional" or "same_group" (only for occupation)
        :param type_node: "occupation", "skill" or "knowledge"
        :param min_: Min number of skills
        :param max_: Max number of skills
        :param exclude: list of nodes to exclude
        :param convert_ids: if true coverts ids into name

        :return: A list of len "max_" with an "n" number of skill "min_" <= "n" <= "max_"
        """
        if exclude is not None:
            exclude = [self.name2id[e] for e in exclude]
        list_ = self.return_neighbors(id_occ, relation, type_node, exclude, convert_ids)

        n = min(random.randint(min_, max_), len(list_))

        sampled = random.sample(list_, n)

        if n < max_:
            sampled.extend(["-" for _ in range(n, max_)])
        return sampled

    def sample_occupation(self) -> tuple[str, str, str]:
        """
        Sample an occupation
        :return: id_occupation, label, isco group
        """
        id_occ = self.occupation.sample().index[0]
        return id_occ, self.graph.nodes[id_occ]["label"], self.graph.nodes[id_occ]["isco_group"]

    def get_job_with_skill(self, competences=None, knowledge=None) -> list[str]:
        """
        Give a list of competence & knowledge, return a list of occupation that has (at least) these skills
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

    def show_subgraph(self, id_occ: str, relation: RelationNode, type_node: TypeNode):
        nodes = list(self.return_neighbors(id_occ, relation, type_node))
        nodes.append(id_occ)

        labels = {node: self.graph.nodes[node]["label"] for node in nodes}

        subgraph = self.graph.subgraph(nodes)
        colors = ["#b35900" if node == id_occ else "#99ccff" for node in subgraph]
        node_size = [500 if node == id_occ else 200 for node in subgraph]

        pos = nx.spring_layout(subgraph)
        plt.figure(figsize=(15, 8))
        nx.draw(subgraph, pos, with_labels=True,
                labels=labels, node_color=colors,
                node_size=node_size)
        plt.show()
