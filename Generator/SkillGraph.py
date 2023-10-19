import itertools
import sys

import networkx as nx
from numpy import zeros, eye


class SkillGraph:
    def __init__(self, jobs_lib: dict):

        # indexing
        all_skill = []
        for job in jobs_lib.values():
            all_skill.extend(job["skills"])

        skills = list(set(all_skill))

        self.__name2id = {skill: id_ for id_, skill in enumerate(skills)}
        self.__id2name = {id_: skill for id_, skill in enumerate(skills)}

        adj_matrix = zeros((len(skills), len(skills)))

        for job in jobs_lib.values():
            for a, b in itertools.product(job["skills"], job["skills"]):
                adj_matrix[self.__name2id[a], self.__name2id[b]] = 1

        # remove self-loop
        adj_matrix -= eye(adj_matrix.shape[0])

        self.__skill_graph = nx.Graph(adj_matrix)

    def shortest_path(self, skill_a: str, skill_b: str, output: str = "len") -> list[str] | None | int:

        if (skill_a not in self.__name2id) or (skill_b not in self.__name2id):
            return sys.maxsize if output == "len" else None

        try:
            shortest_path = nx.shortest_path(self.__skill_graph,
                                             source=self.__name2id[skill_a],
                                             target=self.__name2id[skill_b])

            hop_list = list(map(lambda x: self.__id2name[x], shortest_path))
            return len(hop_list) if output == "len" else hop_list

        except nx.NetworkXNoPath:
            return sys.maxsize if output == "len" else None
