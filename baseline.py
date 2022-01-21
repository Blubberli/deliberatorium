import os
from pathlib import Path

from encode_nodes import MapEncoder
from argumentMap import KialoMap, DeliberatoriumMap
from evaluation import Evaluation


def evaluate_map(encoder_mulitlingual, argument_map, node_types):
    print(argument_map._name)
    encoder_mulitlingual.encode_argument_map(argument_map)
    # default setting: all nodes are evaluated, all nodes are considered as candidates
    eval = Evaluation(argument_map=argument_map)
    mrr = eval.mean_reciprocal_rank(eval._ranks)
    sucess_rate = eval.precision_at_rank(eval._ranks, 5) * 100
    accuracy = eval.precision_at_rank(eval._ranks, 1)
    print("default setting: all nodes are evaluated, all nodes are considered as candidates")
    print("child nodes: %d candidates :%d MRR: %.2f SUCESS: %.2f ACCURACY: %.2f" % (
        len(eval.child_nodes), len(eval._candidate_nodes), mrr, sucess_rate, accuracy))
    # only check for leaf nodes
    eval = Evaluation(argument_map=argument_map, only_leafs=True)
    mrr = eval.mean_reciprocal_rank(eval._ranks)
    sucess_rate = eval.precision_at_rank(eval._ranks, 5) * 100
    accuracy = eval.precision_at_rank(eval._ranks, 1)
    print("only check for leaf nodes")
    print("child nodes: %d candidates :%d MRR: %.2f SUCESS: %.2f ACCURACY: %.2f" % (
        len(eval.child_nodes), len(eval._candidate_nodes), mrr, sucess_rate, accuracy))
    # only leaf nodes and only issues and ideas as parents
    eval = Evaluation(argument_map=argument_map, only_leafs=True, candidate_node_types=node_types)
    mrr = eval.mean_reciprocal_rank(eval._ranks)
    sucess_rate = eval.precision_at_rank(eval._ranks, 5) * 100
    accuracy = eval.precision_at_rank(eval._ranks, 1)
    print("only leaf nodes and only %s as parents" % (",").join(node_types))
    print("child nodes: %d candidates :%d MRR: %.2f SUCESS: %.2f ACCURACY: %.2f" % (
        len(eval.child_nodes), len(eval._candidate_nodes), mrr, sucess_rate, accuracy))


def deliberatorium_baseline():
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/english_maps"
    maps = os.listdir(data_path)

    encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="all-mpnet-base-v2",
                                      normalize_embeddings=True, use_descriptions=False)

    for map in maps:
        argument_map = DeliberatoriumMap("%s/%s" % (str(data_path), map))
        evaluate_map(encoder_mulitlingual, argument_map, {"issue", "idea"})


def kialo_baseline():
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/kialo_maps"
    maps = os.listdir(data_path)
    encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="all-mpnet-base-v2",
                                      normalize_embeddings=True, use_descriptions=False)
    for map in maps:
        print(map)
        path = "%s/%s" % (data_path, map)
        argument_map = KialoMap(path)
        evaluate_map(encoder_mulitlingual, argument_map, {"Pro"})


if __name__ == '__main__':
    kialo_baseline()
