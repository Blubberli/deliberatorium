from argumentMap import ArgumentMap
from encode_nodes import MapEncoder
import numpy as np
import os
import random

random.seed(42)


class Evaluation:

    def __init__(self, argument_map, number_of_candidates=0, node_type=None, restrict_parent_types=False):
        # argument map with encoded nodes
        self._argument_map = argument_map
        # the nodes that are possible candidates to attach the child to
        self._all_candidates = self._argument_map._all_children
        print(len(self._all_candidates))
        # the children to be attached and their correct parents
        self._child_nodes, self._parent_nodes = self.extract_child_and_parent_nodes(node_type)
        if restrict_parent_types:
            # filter out children that have pro or con as parent, these are also not considered as possible candidates
            self._child_nodes, self._parent_nodes, self._all_candidates = self.filter_candidates_for_node_type()
        if number_of_candidates > 0:
            # downsample the number of possible nodes consideres as candidates (e.g. in oder to compare datasets of different sizes)
            self._child_nodes, self._parent_nodes, self._all_candidates = self.filter_candidates_by_number(
                number_of_candidates)
        print(len(self._child_nodes))
        print(len(self._parent_nodes))
        print(len(self._all_candidates))
        assert len(self._child_nodes) == len(
            self._parent_nodes), "the number of children and their parents is not the same"
        self._ranks = self.compute_ranks()

    def compute_ranks(self):
        """
        Method to compute the ranks for the candidate nodes. For each candidate node compute the similarity to the 'gold
        node', in this case the correct parent. Compare the similarity to the parent with the similarities to all other
        nodes and count the number of similarities that are higher (i.e. the number of nodes that are more similar to
        the candidate than the parent=correct node). The rank is the position at which the parent is ranked within the list
        of similar nodes (i.e. if rank=1 then the parent node is the most similar node, if rank=3 then there are 2 nodes
        in the list that are more similar to the candidate than the parent)
        :return:
        """
        ranks = []
        node_matrix, self._node2id = self.get_embedding_matrix()
        child_idxs = [self._node2id[node] for node in self._child_nodes]
        candidate_idxs = [self._node2id[node] for node in self._all_candidates]
        # gather all embeddings for my children
        child_repr = np.take(node_matrix, child_idxs, axis=0)
        # gather all embeddings for candidates
        candidate_repr = np.take(node_matrix, candidate_idxs, axis=0)
        # each row contains a similarity between each child and every possible candidate
        # [candidate_size, children_size]
        candidate2children_similarities = np.dot(candidate_repr, np.transpose(child_repr))
        print(candidate2children_similarities.shape)
        print(candidate2children_similarities)
        for i in range(len(self._child_nodes)):
            # compute the similarity between child and correct parent
            child2parent_similarity = np.dot(self._child_nodes[i]._embedding, self._parent_nodes[i]._embedding)
            # delete the similarity between the child node and itself
            target_sims = np.delete(candidate2children_similarities[:, i], child_idxs[i])
            # the rank is the number of vectors with greater similarity that the one between
            # the target representation and the composed one; no sorting is required, just
            # the number of elements that are more similar
            rank = np.count_nonzero(target_sims > child2parent_similarity) + 1
            ranks.append(rank)
        return ranks

    def filter_candidates_by_number(self, n):
        """
        This method can be used to only use a n-size sample as a pool of possible candidates for the ranking. the pool
        will contain the gold parents. if the number of gold parents is too high, the pool will consist an n-sized sample
        of the parents and the children nodes will be reduces correspondingly. If the number of parents is smaller than n
        additional candidates are sampled from all possible nodes of a map
        :param n:
        :return: the (eventually filtered) list of children, the corresponding parents and the candidates to use for ranking
        """
        assert n <= len(
            self._all_candidates), "your sample size is higher than the number of possible nodes in this map"
        if len(set(self._parent_nodes)) > n:
            # store the reduced number of children and parents
            new_children = []
            new_parents = []
            sampled_parents = random.sample(self._child_nodes, n)
            for i in range(len(self._child_nodes)):
                if self._parent_nodes[i] in set(sampled_parents):
                    new_children.append(self._child_nodes[i])
                    new_parents.append(self._parent_nodes[i])
            # possible candidates are the reduced number of possible parents
            all_candidates = new_parents
        else:
            # take the parents and sample the missing number until n from all candidates
            non_parents = list(set(self._all_candidates).difference(set(self._parent_nodes)))
            additional_candidates = random.sample(non_parents, n - len(set(self._parent_nodes)))
            # possible candidates are parents and an additional random sample
            all_candidates = additional_candidates + self._parent_nodes
            new_children = self._child_nodes
            new_parents = self._parent_nodes
        return new_children, new_parents, all_candidates

    def filter_candidates_for_node_type(self):
        """Filter out all parent nodes and possible candidate nodes that are pros or cons"""
        all_candidates = self._all_candidates
        # no pro and cons as candidates
        all_candidates = [node for node in all_candidates if node._type != "pro" and node._type != "con"]
        # remove pros and cons as parents
        new_children = []
        new_parents = []
        # no children with pros or cons as parent nodes
        for i in range(len(self._child_nodes)):
            if self._parent_nodes[i]._type != "con" and self._parent_nodes[i]._type != "pro":
                new_parents.append(self._parent_nodes[i])
                new_children.append(self._child_nodes[i])
        return new_children, new_parents, all_candidates

    def extract_child_and_parent_nodes(self, node_type):
        """
        Extract the children and their corresponding parent. If node type is given (!=None) only child nodes with
        the given node type are evaluated
        :param node_type: can be None or "pro", "con", "issue", "idea"
        :return: child and their parent nodes [list, list]
        """
        candidates = self._argument_map._all_children
        if node_type:
            candidates = [node for node in candidates if node._type == node_type]
        children = [node for node in candidates if node._parent]
        parents = [node._parent for node in children]
        return children, parents

    def get_embedding_matrix(self):
        """
        Create a node2id index that returns a unique id for each node. create an embedding matrix that contains the
        embedding for each node at the corresponding index
        :return: the embeeding matrix [number_of_nodes, embedding_dim], and the node2index as [dict]
        """
        all_nodes = list(set(self._all_candidates + self._child_nodes + self._parent_nodes))
        print("length all nodes %d" % len(all_nodes))
        target2id = dict(zip(all_nodes, range(len(all_nodes))))
        print(len(target2id))
        matrix = [all_nodes[i]._embedding for i in range(len(all_nodes))]
        return np.array(matrix), target2id

    @staticmethod
    def precision_at_rank(ranks, k):
        """
        Computes the number of times a rank is equal or lower to a given rank.
        :param k: the rank for which the precision is computed
        :return: the precision at a certain rank (float)
        """
        assert k >= 1
        correct = len([rank for rank in ranks if rank <= k])
        return correct / len(ranks)

    @staticmethod
    def mean_reciprocal_rank(ranks):
        """
        Computes the mean reciprocal rank for a list of ranks.
        :param ranks: a list of ranks
        :return: the mean reciprocal rank
        """
        precision_scores = sum([1 / r for r in ranks])
        return precision_scores / len(ranks)


if __name__ == '__main__':
    maps = os.listdir("italian_maps")

    encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="paraphrase-multilingual-MiniLM-L12-v2",
                                      normalize_embeddings=True, use_descriptions=False)

    for map in maps:
        argument_map = ArgumentMap("%s/%s" % ("argument_maps", map))
        print(argument_map._name)
        eval = Evaluation(path_to_map="%s/%s" % ("argument_maps", map),
                          sbert_model_identifier="paraphrase-multilingual-mpnet-base-v2",
                          max_seq_len=200, encoder=encoder_mulitlingual)
        ranks_all = eval._ranks
        ranks_per_type = eval.filter_ranks_by_type()
        print(eval.precision_at_rank(ranks_all, 1), eval.precision_at_rank(ranks_all, 3))
        print("MRR")
        print(eval.mean_reciprocal_rank(ranks_all))

    # map_encoder = MapEncoder(sbert_model_identifier="all-MiniLM-L6-v2", max_seq_len=200, normalize_embeddings=True)
    # map_encoder.encode_argument_map(map)

    # self._encoder.save_embeddings(unique_id=dic["ID"], embeddings=dic["embeddings"], path_to_pckl="embeddings/%s" % path_to_map.split("/")[-1].replace(".json", ""))
    # dic = self._encoder.add_stored_embeddings(self._argument_map, "embeddings/%s" % path_to_map.split("/")[-1].replace(".json", ""))
