import os
from pathlib import Path
from argumentMap import KialoMap
from tqdm.autonotebook import tqdm

LANGS = ['english', 'french', 'german', 'italian', 'other']


def read_data(args):
    data_path = (Path.home() / "data/e-delib/kialo/kialoV2" if args['local'] else
                 Path("/mount/projekte/e-delib/data/kialo/kialoV2"))

    assert args['lang'] in [*LANGS, None]

    # list of maps with no duplicates
    maps = []
    for lang in ([args['lang']] if args['lang'] else LANGS):
        print(f'{lang=}')
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
