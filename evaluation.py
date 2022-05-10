import numpy as np


class Evaluation:

    def __init__(self, argument_map, only_parents=False, only_leafs=False, child_node_type=None,
                 candidate_node_types=None, no_ranks=False, max_candidates=0):
        """
        Computes the number of times a rank is equal or lower to a given rank.
        :param argument_map [ArgumentMap]: an ArgumentMap object with initialized embedding nodes (embeddings have to normalized!)
        :param only_parents [bool]: if true, only parents of child nodes to be evaluated are considered as candidates
        :param only_leafs [bool]: if true, only child nodes that are leafs, i.e. have no children are considered for evaluation
        :param child_node_type [str]: if specified, only child nodes of a given type are considered for evaluation
        :param candidate_node_types [set]: if given, has to be given as a set: only nodes with the speicified node types
        are considered as candidates
        """
        # argument map with encoded nodes
        self.argument_map = argument_map
        # gather all nodes in the map and construct a node2index and an embedding matrix
        self.all_nodes = self.argument_map.all_children
        self.node2id, self.embedding_matrix = self.get_embedding_matrix()
        # extract the nodes to be tested
        self.child_nodes = self.get_child_nodes(only_leafs, child_node_type, candidate_node_types)
        # extract their corresponding parents
        self.parent_nodes = [child.parent for child in self.child_nodes]
        # extract possible candidate (all parents must be within the candidates)
        self.candidate_nodes = self.get_candidate_nodes(only_parents, candidate_node_types)
        self.max_candidates = max_candidates
        assert len(self.child_nodes) == len(
            self.parent_nodes), "the number of children and their parents is not the same"
        if not no_ranks:
            self.ranks = self.compute_ranks()

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
        # compute all possible pairwise similarities
        self.node2node_similarity = np.dot(self.embedding_matrix, np.transpose(self.embedding_matrix))
        # extract child IDs
        self.child_idxs = [self.node2id[node] for node in self.child_nodes]
        self.parent_idx = [self.node2id[node] for node in self.parent_nodes]
        # extract candidate IDs
        self.candidate_idxs = [self.node2id[node] for node in self.candidate_nodes]
        # gather similarities for each child to each of the possible candidate nodes and store them in a new matrix
        self.target_similarity_matrix = np.zeros(shape=[len(self.candidate_idxs), len(self.child_idxs)])
        # a list to store the index of the child in the candidate list
        to_delete = [None] * len(self.child_idxs)
        for i in range(len(self.candidate_idxs)):
            for j in range(len(self.child_idxs)):
                if self.candidate_idxs[i] == self.child_idxs[j]:
                    to_delete[j] = i
                self.target_similarity_matrix[i, j] = self.node2node_similarity[
                    self.candidate_idxs[i], self.child_idxs[j]]

        for i in range(len(self.child_nodes)):
            # compute the similarity between child and correct parent
            child2parent_similarity = np.dot(self.child_nodes[i].embedding, self.parent_nodes[i].embedding)
            # similarities between child and all candidates
            target_sims = self.target_similarity_matrix[:, i]
            # print(np.round(target_sims, decimals=2))
            # remove similaritiy between child and itself (if child was within the candidates)
            if to_delete[i]:
                target_sims = np.delete(target_sims, to_delete[i])
            # remove similarity between child and parent:
            # target_sims = np.delete(target_sims, self.parent_idx[i])
            # the rank is the number of embeddings with greater similarity than the one between
            # the child representation and the parent; no sorting is required, just
            # the number of elements that are more similar
            rank = np.count_nonzero((np.random.choice(target_sims, self.max_candidates)
                                     if self.max_candidates and len(target_sims) > self.max_candidates else
                                     target_sims) > child2parent_similarity) + 1
            ranks.append(rank)
        return ranks

    def get_child_nodes(self, only_leafs, child_node_type, candidate_node_types):
        """Extract the child nodes to be used for evaluation. Apply filtering rules if specified."""
        # case 1: I want to test all possible child nodes in this map
        child_nodes = [node for node in self.all_nodes if node.parent]
        # case 2: I only want to test leaf nodes (= the nodes that were added 'the latest')
        if only_leafs:
            child_nodes = [node for node in child_nodes if node.is_leaf]
        # case 3: I want to test only specific node types
        if child_node_type:
            child_nodes = [node for node in child_nodes if node.type == child_node_type]
        # case 4: I want to test only nodes with certain parent node types
        if candidate_node_types:
            child_nodes = [node for node in child_nodes if node.parent.type in candidate_node_types]
        return child_nodes

    def get_candidate_nodes(self, only_parents, candidate_node_types):
        """Extract the candidate nodes to be used for evaluation. Apply filtering rules if specified."""
        # case 1: consider all nodes of a map as candidates
        candidate_nodes = self.all_nodes
        # case 2: consider only parents as candidates
        if only_parents:
            candidate_nodes = self.parent_nodes
        # filter out candidates of certain types if that is specified
        if candidate_node_types:
            candidate_nodes = [node for node in candidate_nodes if node.type in candidate_node_types]
        return candidate_nodes

    def get_embedding_matrix(self):
        """
        Create a node2id index that returns a unique id for each node. create an embedding matrix that contains the
        embedding for each node at the corresponding index
        :return: the embeeding matrix [number_of_nodes, embedding_dim], and the node2index as [dict]
        """
        target2id = dict(zip(self.all_nodes, range(len(self.all_nodes))))
        matrix = [self.all_nodes[i].embedding for i in range(len(self.all_nodes))]
        return target2id, np.array(matrix)

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
        Computes the mean reciprocal rank for a list of ranks. (As we only have one relevant item, this equals to MAP)
        :param ranks: a list of ranks
        :return: the mean reciprocal rank
        """
        precision_scores = sum([1 / r for r in ranks])
        return precision_scores / len(ranks)
