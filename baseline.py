import os
from pathlib import Path

from encode_nodes import MapEncoder
from argumentMap import KialoMap, DeliberatoriumMap
from evaluation import Evaluation


def evaluate_map(encoder_mulitlingual, argument_map, node_types, max_candidates=0):
    results = {}
    print('eval', argument_map.name)
    encoder_mulitlingual.encode_argument_map(argument_map)
    print("default setting: all nodes are evaluated, all nodes are considered as candidates")
    results['all'] = eval_one(Evaluation(argument_map=argument_map, only_leafs=True, max_candidates=max_candidates))
    print("only check for leaf nodes")
    results['only_leafs'] = eval_one(Evaluation(argument_map=argument_map, only_leafs=True,
                                                max_candidates=max_candidates))
    print("only leaf nodes and only issues and ideas as parents")
    results['only_leafs_limited_types'] = eval_one(Evaluation(argument_map=argument_map, only_leafs=True,
                                                              candidate_node_types=node_types,
                                                              max_candidates=max_candidates))
    return results


def eval_one(evaluation: Evaluation):
    mrr = evaluation.mean_reciprocal_rank(evaluation.ranks)
    p5 = evaluation.precision_at_rank(evaluation.ranks, 5)
    p1 = evaluation.precision_at_rank(evaluation.ranks, 1)
    # print(eval.ranks)
    print("child nodes: %d candidates :%d MRR: %.2f p@5: %.2f p@1: %.2f" % (
        len(evaluation.child_nodes), len(evaluation.candidate_nodes), mrr, p5, p1))
    return {'mrr': mrr, 'p5': p5, 'p1': p1}



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
