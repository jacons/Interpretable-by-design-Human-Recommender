import itertools
import random
from enum import Enum
from functools import reduce

import networkx as nx
from matplotlib import pyplot as plt
from pandas import read_csv


class RelationNode(Enum):
    ES = "essential"
    OP = "optional"
    JB = "same_group"
    AL = "All"


class TypeNode(Enum):
    OC = "occupation"
    KN = "knowledge"
    SK = "skill/competence"
    AL = "all"


class JobGraph:
    OCCUPATION_GROUP_THRESHOLD = 4  # A constant for occupation group threshold

    def __init__(self, occ2skill: str, occupation: str, skills: str):
        """
        Occupation/Skill/Knowledge Graph
        :param occ2skill: occupations - skill relationship (many to many)
        :param occupation: occupation list
        :param skills: skill list
        """
        self.occ2skills = read_csv(occ2skill)  # relation "many to many"
        self.occupation = read_csv(occupation, index_col=0)  # all occupation
        self.skills = read_csv(skills, index_col=0)  # all skill

        # remove an occupation that hasn't "essential skills"
        self.occupation = self.occupation[self.occupation.index != "a580e79a-b752-49c1-b033-b5ab2b34bfba"]
        self.occupation["group"] = self.occupation["group"].str[:self.OCCUPATION_GROUP_THRESHOLD].astype(int)

        self.graph = nx.Graph()

        # Add occupation nodes to the graph
        self.graph.add_nodes_from(
            [(occupation[0], {"type": "occupation", "label": occupation[1], "isco_group": occupation[2]})
             for occupation in self.occupation.itertuples()]
        )

        # Add skill nodes to the graph
        self.graph.add_nodes_from(
            [(skill[0], {"type": skill[2], "label": skill[1], "sector": skill[3]})
             for skill in self.skills.itertuples()]
        )

        # Add edges for occ2skills relations
        self.graph.add_edges_from(
            [(row[1], row[3], {"relation": row[2]}) for row in self.occ2skills.itertuples()]
        )

        # Link together the occupation with the same group
        for (occ1, occ2) in itertools.product(self.occupation.itertuples(), self.occupation.itertuples()):
            if (occ1[0] != occ2[0]) and (occ1[2] == occ2[2]):
                self.graph.add_edge(occ1[0], occ2[0], relation="same_group")

        self.name2id_skill = {tuple_[1]: tuple_[0] for tuple_ in self.skills.itertuples()}

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
        :param exclude: list of node to exclude
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
                      exclude: list = None):
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
        list_ = self.return_neighbors(id_occ, relation, type_node, exclude, convert_ids)

        n = min(random.randint(min_, max_), len(list_))

        sampled = random.sample(list_, n)

        if n < max_:
            sampled.extend(["-" for _ in range(n, max_)])
        return sampled

    def sample_occupation(self) -> tuple[str, str, int]:
        """
        Sample an occupation
        :return: id_occupation, label, isco group
        """
        id_occ = self.occupation.sample().index[0]
        return id_occ, self.graph.nodes[id_occ]["label"], self.graph.nodes[id_occ]["isco_group"]

    def get_similar_job(self, id_occ: str, exclude: list = None, convert_ids: bool = False) -> list:
        """
        Give an occupation return a list of occupation with the same group
        :param id_occ: id occupation
        :param exclude: list of id occupation to exclude
        :param convert_ids: if true convert id into name
        :return:
        """
        return list(self.return_neighbors(id_occ, RelationNode.JB, TypeNode.OC, exclude, convert_ids))

    def get_job_with_skill(self, competences: list[str], knowledge: list[str]) -> list[str]:
        """
        Give a list of competence & knowledge, return a list of occupation that has these skills
        :return:
        """
        skills = competences + knowledge
        nodes = [set(self.graph.neighbors(self.name2id_skill[skill])) for skill in skills]
        if nodes:
            similar_jobs = reduce(set.intersection, nodes)
            return list(similar_jobs)
        else:
            return []

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
