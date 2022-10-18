from analysis.base import get_experiment_path

from kialo_util import read_data

detailed = False
top_k = 1


experiments = ['new/all-mpnet-zero',
               'new/all-mpnet-mulneg',
               'new/all-mpnet-mulneg-beginning',
               'new/all-mpnet-mulneg-pro_con',
               ]

maps = read_data({'local': True, 'debug_maps_size': None, 'lang': None})

for experiment in experiments:
    experiment_path = get_experiment_path(experiment)
    # print(experiment_path)

    # ### Pro/Con Statistics

    node_type_stats = {None: [], 1: [], -1: []}
    for m in maps:
        for x in m.child_nodes:
            node_type_stats[None].append(x)
            node_type = x.type
            node_type_stats[node_type].append(x)

    {k: (len(v), len(v) / len(node_type_stats[None])) for k, v in node_type_stats.items()}

    # ## Analyse Node Results

    import json

    all_nodes_results = json.loads((experiment_path / 'results/all/all_nodes.json').read_text())
    argument_maps_dict = {str(x.id): {n.id: n for n in x.all_nodes} for x in maps}

    all_nodes_results['2223']['only_leafs_limited_types'][0]

    # ### Pro/Con

    from collections import defaultdict

    node_type_results = {None: [], 1: [], -1: []}
    maps_node_type_results = {None: defaultdict(list), 1: defaultdict(list), -1: defaultdict(list)}
    for map_id, v in all_nodes_results.items():
        if v['only_leafs_limited_types']:
            for x in v['only_leafs_limited_types']:
                r = (x['rank'], x['id'])
                node_type_results[None].append(r)
                maps_node_type_results[None][map_id].append(r)
                node_type = argument_maps_dict[map_id][x['id']].type
                node_type_results[node_type].append(r)
                maps_node_type_results[node_type][map_id].append(r)

    # from evaluation import Evaluation
    # {k:Evaluation.precision_at_rank(v, 5) for k,v in node_type_results.items()}

    import statistics


    def pak(l, k):
        return statistics.mean(x[0] <= k for x in l)

    if detailed:
        print(' & '.join([str(len(v)) for v in node_type_results.values()]))
        print(' & '.join(
            [f'{statistics.mean(pak(l, top_k) for l in v.values()):.2f}' for k, v in maps_node_type_results.items()]))
        print(' & '.join([f'{pak(v, top_k):.2f}' for k, v in node_type_results.items()]))

    # ## Annotated Samples Results

    import json

    all_samples_results = json.loads((experiment_path / 'results/annotated_samples_predictions.json').read_text())
    all_samples_results['17763.90']

    # ### Pro/Con

    node_type_sample_results = {None: [], 1: [], -1: []}
    for idx, v in all_samples_results.items():
        r = (v['rank'], idx)
        node_type_sample_results[None].append(r)
        node_type_sample_results[v['nodeType']].append(r)

    if detailed:
        print(' & '.join([str(k) for k, v in node_type_results.items()]), end=' & ')
        print(' & '.join([str(k) for k, v in maps_node_type_results.items()]))

        print(' & '.join([str(len(v)) for v in node_type_results.values()]), end=' & ')
        print(' & '.join([str(len(v)) for v in node_type_sample_results.values()]))

    print(' \t '.join(
        [f'{statistics.mean(pak(l, top_k) for l in v.values()):.4f}' for k, v in maps_node_type_results.items()]),
        end=' \t ')
    print(' \t '.join([f'{pak(v, top_k):.3f}' for k, v in node_type_sample_results.items()]))
