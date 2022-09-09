import json
import os

import pandas as pd

from pathlib import Path
from argumentMap import KialoMap
from tqdm.autonotebook import tqdm

LANGS = ['english', 'french', 'german', 'italian', 'other']


def read_data(args):
    data_path = get_base_data_path(args['local']) / 'kialoV2'

    assert args['lang'] in [*LANGS, None]

    # list of maps with no duplicates
    maps = []
    for lang in ([args['lang']] if args['lang'] else LANGS):
        print(f'{lang=}')
        maps += [x for x in data_path.glob(f'{lang}/*.pkl') if x.stem not in [y.stem for y in maps]]

    if args['debug_maps_size']:
        maps = sorted(maps, key=os.path.getsize)
        if args['debug_map_index']:
            maps = list(data_path.glob(f"**/{args['debug_map_index']}.pkl")) + \
                   [x for x in maps if x.stem != args['debug_map_index']]
        maps = maps[:args['debug_maps_size']]

    argument_maps = [KialoMap(str(_map), _map.stem) for _map in tqdm(maps, f'processing maps')
                     # some maps seem to be duplicates with (1) in name
                     if '(1)' not in _map.stem]
    print(f'remaining {len(maps)} maps after clean up')
    return argument_maps


def read_annotated_maps_ids(local: bool):
    data_path = get_annotation_data_path(local)
    annotated_maps_df = pd.read_csv(data_path / 'all_maps.csv', sep='\t')
    ids = annotated_maps_df['mapID'].to_list()
    return ids


def read_annotated_samples(local: bool, args: dict = None):
    data_path = get_annotation_data_path(local)
    data = json.loads((data_path / 'target_and_candidate_info.json').read_text())
    # clean up instances where the child node is in candidates
    for node_id, sample in data.items():
        if node_id in sample['candidates']:
            print(f'removing {node_id} from its own candidates')
            del sample['candidates'][node_id]
    if args and args['debug_maps_size']:
        data = {k: v for k, v in list(data.items())[:args['debug_maps_size']]}
    return data


def get_annotation_data_path(local: bool):
    return get_base_data_path(local) / 'annotation/annotation200instances'


def get_base_data_path(local: bool):
    return (Path.home() / "data/e-delib/kialo" if local else
            Path("/mount/projekte/e-delib/data/kialo"))

