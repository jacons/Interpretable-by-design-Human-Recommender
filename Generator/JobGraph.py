import random

import networkx as nx
from matplotlib import pyplot as plt
from pandas import read_csv

"""
For each occupation there is at least one essential skill, but may happen
that it has zero optional skills.
"""


class JobGraph:
    def __init__(self, occ2skill: str, occupation: str, skills: str):

        self.occ2skills = read_csv(occ2skill)  # relation "many to many"
        self.occupation = read_csv(occupation)  # all occupation
        self.skills = read_csv(skills)  # all skill

        # remove an occupation that hasn't "essential skills"
        self.occupation = self.occupation[self.occupation["id_occupation"] != "a580e79a-b752-49c1-b033-b5ab2b34bfba"]

        # dictionary id_skill to name
        self.ids_kill2name = {tuple_[1]: tuple_[2] for tuple_ in self.skills.itertuples()}
        # dictionary id_occupation to name
        self.id_occ2name = {tuple_[1]: tuple_[2] for tuple_ in self.occupation.itertuples()}

        self.graph_essential = nx.Graph()
        self.graph_optional = nx.Graph()

        # Add all occupations in the graph
        for occupation in self.occupation["id_occupation"]:
            self.graph_essential.add_node(occupation)
            self.graph_optional.add_node(occupation)

        dt = self.occ2skills[self.occ2skills["relationType"] == "essential"]
        for i in dt.itertuples():
            self.graph_essential.add_edge(i[1], i[3])

        dt = self.occ2skills[self.occ2skills["relationType"] == "optional"]
        for i in dt.itertuples():
            self.graph_optional.add_edge(i[1], i[3])

    def sample_skills(self, id_occ: str, relation_type: str, min_: int = 2, max_: int = 6, convert_name: bool = False):

        if relation_type == "essential":
            list_ = list(self.graph_essential.neighbors(id_occ))
        else:
            list_ = list(self.graph_optional.neighbors(id_occ))

        if convert_name:
            list_ = [self.ids_kill2name[element] for element in list_]
        n = min(random.randint(min_, max_), len(list_))

        sampled = random.sample(list_, n)

        if n < max_:
            sampled.extend(["-" for _ in range(n, max_)])
        return sampled

    def show_skills(self, id_occ: str, relation_type):

        graph = self.graph_essential if relation_type == "essential" else self.graph_optional

        nodes = list(graph.neighbors(id_occ))
        labels = {node: self.ids_kill2name[node] for node in nodes}
        nodes.extend([id_occ])
        labels[id_occ] = self.id_occ2name[id_occ]

        subgraph = graph.subgraph(nodes)
        colors = ["#b35900" if node == id_occ else "#99ccff" for node in subgraph]
        node_size = [500 if node == id_occ else 200 for node in subgraph]

        pos = nx.spring_layout(subgraph)
        plt.figure(figsize=(15, 8))
        nx.draw(subgraph, pos, with_labels=True,
                labels=labels, node_color=colors,
                node_size=node_size)
        plt.show()

    def sample_occupation(self, convert_name: bool = False):
        occupation = self.occupation["id_occupation"].sample().values[0]

        if convert_name:
            return occupation, self.id_occ2name[occupation]
        else:
            return occupation

