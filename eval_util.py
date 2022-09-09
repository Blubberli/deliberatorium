import logging

from evaluation import Evaluation
from rerank_evaluation import RerankEvaluation

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
    metrics = Evaluation.calculate_metrics(evaluation.ranks)
    metrics['dist'] = evaluation.average_taxonomic_distance(0.5)
    # print(eval.ranks)
    print(f"child nodes: {len(evaluation.child_nodes)} candidates :{len(evaluation.candidate_nodes)}. " +
          format_metrics(metrics))
    return metrics, \
           [(c.id, r, p.id, t) for c, r, p, t in zip(evaluation.child_nodes, evaluation.ranks, evaluation.predictions,
                                                     evaluation.taxonomic_distances)]


def create_evaluation(eval_args: dict):
    evaluation_class = RerankEvaluation if 'cross_encoder' in eval_args else Evaluation
    return evaluation_class(**eval_args)


def format_metrics(metrics):
    return ' , '.join([f"{k}: {v:.2f}" for k, v in metrics.items()])