import os
import re

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
    data_path = get_base_data_path(local) / 'annotation/annotation200instances'
    annotated_maps_df = pd.read_csv(data_path / 'all_maps.csv', sep='\t')
    ids = annotated_maps_df['mapID'].to_list()
    return ids


def get_base_data_path(local: bool):
    return (Path.home() / "data/e-delib/kialo" if local else
            Path("/mount/projekte/e-delib/data/kialo"))


def remove_url_and_hashtags(text):
    pattern = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    match = re.findall(pattern, text)
    for m in match:
        url = m[0]
        text = text.replace(url, '')
    text = text.replace("()", "")
    text = text.replace("#", "")
    return text
