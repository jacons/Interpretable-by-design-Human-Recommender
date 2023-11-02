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
    def __init__(self, occ2skill: str, occupation: str, skills: str):
        self.occ2skills = read_csv(occ2skill)  # relation "many to many"
        self.occupation = read_csv(occupation, index_col=0)  # all occupation
        self.skills = read_csv(skills, index_col=0)  # all skill

        # remove an occupation that hasn't "essential skills"
        self.occupation = self.occupation[self.occupation.index != "a580e79a-b752-49c1-b033-b5ab2b34bfba"]
        self.occupation["group"] = self.occupation["group"].str[:4]

        self.graph = nx.Graph()
        for occupation in self.occupation.itertuples():
            self.graph.add_node(occupation[0], type="occupation", label=occupation[1])

        for skill in self.skills.itertuples():
            self.graph.add_node(skill[0], type=skill[2], label=skill[1])

        for relation in self.occ2skills.itertuples():
            self.graph.add_edge(relation[1], relation[3], relation=relation[2])

        for (occ1, occ2) in itertools.product(self.occupation.itertuples(), self.occupation.itertuples()):
            if (occ1[0] != occ2[0]) and (occ1[2] == occ2[2]):
                self.graph.add_edge(occ1[0], occ2[0], relation="same_group")

        self.name2id_skill = {tuple_[2]: tuple_[1] for tuple_ in self.skills.itertuples()}

    def return_neighbors(self, id_occ: str,
                         relation: RelationNode,
                         type_node: TypeNode,
                         exclude: list = None,
                         convert_ids: bool = False) -> filter | map:
        if exclude is None:
            exclude = []

        filtered_skills = filter(
            lambda n: self.graph.edges[id_occ, n]["relation"] == relation.value and
                      self.graph.nodes[n]["type"] == type_node.value and
                      n not in exclude,
            self.graph.neighbors(id_occ))

        if convert_ids:
            filtered_skills = map(lambda x: self.graph.nodes[x]["label"], filtered_skills)
        return filtered_skills

    def sample_skills(self,
                      id_occ: str,
                      relation: RelationNode,
                      type_node: TypeNode,
                      min_: int = 2, max_: int = 6,
                      convert_name: bool = False,
                      exclude: list = None):

        list_ = list(self.return_neighbors(id_occ, relation, type_node, exclude, convert_name))

        n = min(random.randint(min_, max_), len(list_))

        sampled = random.sample(list_, n)

        if n < max_:
            sampled.extend(["-" for _ in range(n, max_)])
        return sampled

    def sample_occupation(self, convert_ids: bool = False):
        occ = self.occupation.sample()["id_occupation"]
        if convert_ids:
            return self.graph.nodes[occ]["label"]

    def get_similar_job(self, id_occ: str, exclude: list = None, convert_ids: bool = False) -> list:
        return list(self.return_neighbors(id_occ, RelationNode.JB, TypeNode.OC, exclude, convert_ids))

    def get_job_with_skill(self, skills: list) -> list:

        nodes = [set(self.graph.neighbors(self.name2id_skill[skill])) for skill in skills]
        similar_jobs = reduce(lambda x, y: x.intersection(y), nodes)
        return list(similar_jobs)

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
