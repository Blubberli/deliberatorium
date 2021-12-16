import os
from pathlib import Path

from encode_nodes import MapEncoder
from argumentMap import ArgumentMap
from evaluation import Evaluation


def evaluate_map(encoder_mulitlingual, argument_map):
    print(argument_map._name)
    encoder_mulitlingual.encode_argument_map(argument_map)
    print(argument_map._name)
    # default setting: all nodes are evaluated, all nodes are considered as candidates
    eval = Evaluation(argument_map=argument_map)
    mrr = eval.mean_reciprocal_rank(eval._ranks)
    sucess_rate = eval.precision_at_rank(eval._ranks, 5) * 100
    # print(eval._ranks)
    print("default setting: all nodes are evaluated, all nodes are considered as candidates")
    print("child nodes: %d candidates :%d MRR: %.2f SUCESS: %.2f" % (
        len(eval.child_nodes), len(eval._candidate_nodes), mrr, sucess_rate))
    # only check for leaf nodes
    eval = Evaluation(argument_map=argument_map, only_leafs=True)
    mrr = eval.mean_reciprocal_rank(eval._ranks)
    sucess_rate = eval.precision_at_rank(eval._ranks, 5) * 100
    print("only check for leaf nodes")
    print("child nodes: %d candidates :%d MRR: %.2f SUCESS: %.2f" % (
        len(eval.child_nodes), len(eval._candidate_nodes), mrr, sucess_rate))
    # only leaf nodes and only issues and ideas as parents
    eval = Evaluation(argument_map=argument_map, only_leafs=True, candidate_node_types={"issue", "idea"})
    mrr = eval.mean_reciprocal_rank(eval._ranks)
    sucess_rate = eval.precision_at_rank(eval._ranks, 5) * 100
    print("only leaf nodes and only issues and ideas as parents")
    print("child nodes: %d candidates :%d MRR: %.2f SUCESS: %.2f" % (
        len(eval.child_nodes), len(eval._candidate_nodes), mrr, sucess_rate))


def main():
    data_path = Path.home() / "data/e-delib/deliberatorium/maps/english_maps"
    maps = os.listdir(data_path)

    encoder_mulitlingual = MapEncoder(max_seq_len=128, sbert_model_identifier="all-mpnet-base-v2",
                                      normalize_embeddings=True, use_descriptions=False)

    for map in maps:
        argument_map = ArgumentMap("%s/%s" % (str(data_path), map))
        evaluate_map(encoder_mulitlingual, argument_map)


if __name__ == '__main__':
    main()