from argumentMap import KialoMap
from evaluation import Evaluation
from encode_nodes import MapEncoder
import numpy as np
import unittest


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.small_map = KialoMap(
            data_path="kialo_maps/dogs-do-not-have-complex-emotions-such-as-shame-guilt-and-pride-31662.txt")
        self.encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="all-mpnet-base-v2",
                                               normalize_embeddings=True, use_descriptions=False)
        self.encoder_mulitlingual.encode_argument_map(self.small_map)

        # the root nodes
        self.first_level_nodes = self.small_map.direct_children
        self.evaluation_all_types = Evaluation(self.small_map, only_leafs=True)
        self.evaluation_only_con = Evaluation(self.small_map, only_leafs=True, child_node_type="Con")
        self.evaluation_only_pro = Evaluation(self.small_map, only_leafs=True, child_node_type="Pro")
        self.evaluation_all_close_relatives = Evaluation(self.small_map, only_leafs=True, close_relatives=True)

    def test_number_of_nodes_all_types(self):
        candidates = self.evaluation_all_types.candidate_nodes
        child_nodes = self.evaluation_all_types.child_nodes
        parent_nodes = self.evaluation_all_types.parent_nodes
        np.testing.assert_equal(len(candidates), 16)
        np.testing.assert_equal(len(child_nodes), 7)
        np.testing.assert_equal(len(parent_nodes), 7)

    def test_number_of_nodes_type_based(self):
        # if I only want to test "con" nodes there should be 4
        np.testing.assert_equal(len(self.evaluation_only_con.child_nodes), 4)
        # if I only want to test "pro" nodes there should be 3
        np.testing.assert_equal(len(self.evaluation_only_pro.child_nodes), 3)

    def test_similarity_matrix(self):
        # the target similarity matrix should have the shape candidates x children
        np.testing.assert_equal(self.evaluation_all_types.target_similarity_matrix.shape[0], 16)
        np.testing.assert_equal(self.evaluation_all_types.target_similarity_matrix.shape[1], 7)
        np.testing.assert_equal(self.evaluation_only_con.target_similarity_matrix.shape[0], 16)
        np.testing.assert_equal(self.evaluation_only_con.target_similarity_matrix.shape[1], 4)

        # test whether the similarity between a node and itself is almost 1.0
        # node at index 3 is a child and a candidate
        np.testing.assert_almost_equal(self.evaluation_all_types.target_similarity_matrix[3][0], 1.0, decimal=2)

    def test_ranks(self):
        # rank | node | parent | node2node sim | nodes with higher sim
        # 1        3     2          0.757           0
        # 3        4     2          0.571           2
        # 2        8     7          0.734           1
        # 3        10    9          0.519           2
        # 2        11    9          0.638           1
        # 2        12    9          0.613           1
        # 11       15    14         0.207           10
        np.testing.assert_almost_equal(self.evaluation_all_types.ranks, [1, 3, 2, 3, 2, 2, 11])
        np.testing.assert_almost_equal(self.evaluation_only_con.ranks, [3, 2, 2, 11])
        np.testing.assert_almost_equal(self.evaluation_only_pro.ranks, [1, 3, 2])

    def test_metrics(self):
        # ranks: [1, 3, 2, 3, 2, 2, 11])
        # precision_rank1 = 1/7 = 0.14; precision_rank5 = 6/7 = 0.86; MRR: 1/1 + 2*1/3 + 3*1/2 + 1/11 = 3.258 / 7 = 0.47
        prec1 = 0.14
        prec5 = 0.86
        mrr = 0.47
        ranks = self.evaluation_all_types.ranks
        np.testing.assert_almost_equal(self.evaluation_all_types.precision_at_rank(ranks, 1), prec1, decimal=2)
        np.testing.assert_almost_equal(self.evaluation_all_types.precision_at_rank(ranks, 5), prec5, decimal=2)
        np.testing.assert_almost_equal(self.evaluation_all_types.mean_reciprocal_rank(ranks), mrr, decimal=2)

    def test_close_relatives(self):
        # ranks = [1, 1, 2, 1, 2, 1, 11]
        # prec1 = 4/7 = 0.57; prec5 = 6/7 = 0.86; MRR: 4+0.5+0.5+1/11 = 5.09 / 7 = 0.73
        prec1 = 0.57
        prec5 = 0.86
        mrr = 0.73
        ranks = [1, 1, 2, 1, 2, 1, 11]
        np.testing.assert_almost_equal(self.evaluation_all_close_relatives.ranks, ranks)
        np.testing.assert_almost_equal(self.evaluation_all_close_relatives.precision_at_rank(ranks, 1), prec1,
                                       decimal=2)
        np.testing.assert_almost_equal(self.evaluation_all_close_relatives.precision_at_rank(ranks, 5), prec5,
                                       decimal=2)
        np.testing.assert_almost_equal(self.evaluation_all_close_relatives.mean_reciprocal_rank(ranks), mrr, decimal=2)

    def test_taxonomic_distance(self):
        nodes = self.small_map.all_children
        node1 = nodes[0]
        node2 = nodes[3]
        # distance should be 4
        np.testing.assert_equal(node1.shortest_path(node2), 4)
        node2 = nodes[5]
        # distance should be 2
        np.testing.assert_equal(node1.shortest_path(node2), 2)
        # distance shouldbe 0
        np.testing.assert_equal(node1.shortest_path(node1), 0)
        # distance should be 3
        node2 = nodes[10]
        np.testing.assert_equal(node1.shortest_path(node2), 3)
        node1 = nodes[3]
        np.testing.assert_equal(node1.shortest_path(node2), 5)

    def test_average_distance(self):
        # q1 = 1; q2 = 1; q3 =
        q1 = self.evaluation_all_types.average_taxonomic_distance(0.25)
        q2 = self.evaluation_all_types.average_taxonomic_distance(0.50)
        q3 = self.evaluation_all_types.average_taxonomic_distance(0.75)

        print(q1)
