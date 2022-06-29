from argumentMap import KialoMap
from evaluation import Evaluation
from encode_nodes import MapEncoder
import numpy as np
import unittest


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        from pathlib import Path
        self.small_map = KialoMap(
            data_path=(str(Path.home() / "data/e-delib/kialo/kialoV2/unittest_data/10133.pkl")))
        self.encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="all-mpnet-base-v2",
                                               normalize_embeddings=True, use_descriptions=False)
        self.encoder_mulitlingual.encode_argument_map(self.small_map)

        # the root nodes
        self.first_level_nodes = self.small_map.direct_children
        self.evaluation_all_types = Evaluation(self.small_map, only_leafs=True)
        self.evaluation_only_con = Evaluation(self.small_map, only_leafs=True, child_node_type=-1)
        self.evaluation_only_pro = Evaluation(self.small_map, only_leafs=True, child_node_type=1)
        self.evaluation_all_close_relatives = Evaluation(self.small_map, only_leafs=True, close_relatives=True)

    def test_number_of_nodes_all_types(self):
        candidates = self.evaluation_all_types.candidate_nodes
        child_nodes = self.evaluation_all_types.child_nodes
        parent_nodes = self.evaluation_all_types.parent_nodes
        np.testing.assert_equal(len(candidates), 14)
        np.testing.assert_equal(len(child_nodes), 4)
        np.testing.assert_equal(len(parent_nodes), 4)

    def test_number_of_nodes_type_based(self):
        # if I only want to test "con" nodes there should be 4
        np.testing.assert_equal(len(self.evaluation_only_con.child_nodes), 0)
        # if I only want to test "pro" nodes there should be 3
        np.testing.assert_equal(len(self.evaluation_only_pro.child_nodes), 4)

    def test_similarity_matrix(self):
        # the target similarity matrix should have the shape candidates x children
        np.testing.assert_equal(self.evaluation_all_types.target_similarity_matrix.shape[0], 14)
        np.testing.assert_equal(self.evaluation_all_types.target_similarity_matrix.shape[1], 4)
        np.testing.assert_equal(self.evaluation_only_pro.target_similarity_matrix.shape[0], 14)
        np.testing.assert_equal(self.evaluation_only_pro.target_similarity_matrix.shape[1], 4)

        # test whether the similarity between a node and itself is almost 1.0
        # node at index 3 is a child and a candidate
        np.testing.assert_almost_equal(self.evaluation_all_types.target_similarity_matrix[6][0], 1.0, decimal=2)

    def test_ranks(self):
        # rank | node | parent | node2node sim | nodes with higher sim | predicted parent
        # 12       6     5          0.432           11                      7
        # 2        8     7          0.568           1                       12
        # 1        10    9          0.457           0                       9
        # 1        12    11         0.714           0                       11
        np.testing.assert_almost_equal(self.evaluation_all_types.ranks, [12, 2, 1, 1])
        np.testing.assert_almost_equal(self.evaluation_only_con.ranks, [])
        np.testing.assert_almost_equal(self.evaluation_only_pro.ranks, [12, 2, 1, 1])

    def test_metrics(self):
        # ranks: [12, 2, 1, 1])
        # precision_rank1 = 2/4 = 0.50; precision_rank5 = 3/4 = 0.75; MRR: 1/12 + 2*1/1 + 1/2 = 2.583 / 4 = 0.646
        prec1 = 0.50
        prec5 = 0.75
        mrr = 0.646
        ranks = self.evaluation_all_types.ranks
        np.testing.assert_almost_equal(self.evaluation_all_types.precision_at_rank(ranks, 1), prec1, decimal=2)
        np.testing.assert_almost_equal(self.evaluation_all_types.precision_at_rank(ranks, 5), prec5, decimal=2)
        np.testing.assert_almost_equal(self.evaluation_all_types.mean_reciprocal_rank(ranks), mrr, decimal=2)

    def test_close_relatives(self):
        # ranks = [12, 2, 1, 1]
        # dist(predicted parents)  [2, 5, 1, 1]
        # prec1 = 3/4 = 0.75; prec5 = 2/4 = 0.75; MRR: (1/2+3) / 4 = 0.876
        prec1 = 0.75
        prec5 = 1.0
        mrr =  0.876
        ranks = [1, 2, 1, 1]
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
        # distance should be 2
        np.testing.assert_equal(node1.shortest_path(node2), 2)
        node2 = nodes[5]
        # distance should be 2
        np.testing.assert_equal(node1.shortest_path(node2), 2)
        # distance shouldbe 0
        np.testing.assert_equal(node1.shortest_path(node1), 0)
        # distance should be 3
        node2 = nodes[10]
        np.testing.assert_equal(node1.shortest_path(node2), 3)
        node1 = nodes[8]
        np.testing.assert_equal(node1.shortest_path(node2), 5)
        node2 = nodes[1]
        np.testing.assert_equal(node1.shortest_path(node2), 4)

    def test_average_distance(self):
        # q1 = 1; q2 = 1.5; q3 = 3.5;
        q1 = self.evaluation_all_types.average_taxonomic_distance(0.25)
        q2 = self.evaluation_all_types.average_taxonomic_distance(0.50)
        q3 = self.evaluation_all_types.average_taxonomic_distance(0.75)
        np.testing.assert_equal(q1, 1)
        np.testing.assert_equal(q2, 1.5)
        np.testing.assert_equal(q3, 3.5)


if __name__ == '__main__':
    unittest.main()