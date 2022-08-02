import os
import re
from pathlib import Path
from argumentMap import KialoMap
from tqdm.autonotebook import tqdm


def read_data(args):
    data_path = (Path.home() / "data/e-delib/kialo/kialoV2" if args['local'] else
                 Path("/mount/projekte/e-delib/data/kialo/kialoV2"))
    # list of maps with no duplicates
    maps = []
    for lang in ['english', 'french', 'german', 'italian', 'other']:
        maps += [x for x in data_path.glob(f'{lang}/*.pkl') if x.stem not in [y.stem for y in maps]]

    if args['debug_maps_size']:
        maps = sorted(maps, key=os.path.getsize)
        if args['debug_map_index']:
            maps = list(data_path.glob(f"**/{args['debug_map_index']}.pkl")) + maps
        maps = maps[:args['debug_maps_size']]

    argument_maps = [KialoMap(str(_map), _map.stem) for _map in tqdm(maps, f'processing maps')
                     # some maps seem to be duplicates with (1) in name
                     if '(1)' not in _map.stem]
    print(f'remaining {len(maps)} maps after clean up')
    return argument_maps


def remove_url_and_hashtags(text):
    pattern = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    match = re.findall(pattern, text)
    for m in match:
        url = m[0]
        text = text.replace(url, '')
    text = text.replace("()", "")
    text = text.replace("#", "")
    return text