import logging

import os
from pathlib import Path

from encode_nodes import MapEncoder
from argumentMap import KialoMap, DeliberatoriumMap
from evaluation import Evaluation
from sentence_transformers import LoggingHandler

from rerank_evaluation import RerankEvaluation

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

METRICS = ['mrr', 'p5', 'p1', 'dist']


def evaluate_map(encoder_mulitlingual, argument_map, node_types, max_candidates=0, cross_encoder=None):
    results = {}
    node_results = {}
    print(f'eval {argument_map.id} {len(argument_map.all_children)=} \n{argument_map.name}')
    encoder_mulitlingual.encode_argument_map(argument_map)

    print("default setting: all nodes are evaluated, all nodes are considered as candidates")
    eval_args = {'argument_map': argument_map, 'only_leafs': False, 'max_candidates': max_candidates}
    if cross_encoder:
        eval_args['cross_encoder'] = cross_encoder
    results['all'], node_results['all'] = eval_one(eval_args)

    print("only check for leaf nodes")
    eval_args['only_leafs'] = True
    results['only_leafs'], node_results['only_leafs'] = eval_one(eval_args)

    print("only leaf nodes and only issues and ideas as parents")
    eval_args['candidate_node_types'] = node_types
    results['only_leafs_limited_types'], node_results['only_leafs_limited_types'] = eval_one(eval_args)

    return results, node_results


def eval_one(eval_args: dict):
    evaluation = create_evaluation(eval_args)
    if len(evaluation.child_nodes) == 0:
        logging.warning('no child nodes found')
        return None, None
    mrr = evaluation.mean_reciprocal_rank(evaluation.ranks)
    p5 = evaluation.precision_at_rank(evaluation.ranks, 5)
    p1 = evaluation.precision_at_rank(evaluation.ranks, 1)
    dist = evaluation.average_taxonomic_distance(0.5)
    # print(eval.ranks)
    print("child nodes: %d candidates :%d MRR: %.2f p@5: %.2f p@1: %.2f dist: %.2f" % (
        len(evaluation.child_nodes), len(evaluation.candidate_nodes), mrr, p5, p1, dist))
    return dict(zip(METRICS, [mrr, p5, p1, dist])), \
           [(c.id, r, p.id, t) for c, r, p, t in zip(evaluation.child_nodes, evaluation.ranks, evaluation.predictions,
                                                     evaluation.taxonomic_distances)]


def create_evaluation(eval_args: dict):
    evaluation_class = RerankEvaluation if 'cross_encoder' in eval_args else Evaluation
    return evaluation_class(**eval_args)



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
