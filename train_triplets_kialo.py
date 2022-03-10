import faulthandler
import itertools
import json
import logging
import math
import os
import random
import signal
from pathlib import Path

import wandb
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers import models, losses, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from argumentMap import KialoMap
from baseline import evaluate_map
from encode_nodes import MapEncoder
from evaluation import Evaluation
from kialo_domains_util import get_maps2uniquetopic
from train_triplets_delib import parse_args, get_model_save_path

AVAILABLE_MAPS = ['dopariam1', 'dopariam2', 'biofuels', 'RCOM', 'CI4CG']

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def add_more_args(parser):
    parser.add_argument('--training_domain_index', type=int, default=-1)


def main():
    faulthandler.register(signal.SIGUSR1.value)

    args = parse_args(add_more_args)

    model_name = args['model_name_or_path']
    train_batch_size = args['train_batch_size']  # The larger you select this, the better the results (usually)
    max_seq_length = 75
    num_epochs = args['num_train_epochs']

    data_path = (Path.home() / "data/e-delib/kialo/kialo_maps" if args['local'] else
                 Path("/mount/projekte/e-delib/data/kialo/kialo_maps"))
    maps = list(data_path.glob(f"*.txt"))

    if args['debug_maps_size']:
        maps = sorted(maps, key=os.path.getsize)
        maps = maps[:args['debug_maps_size']]

    argument_maps = [KialoMap(str(_map), _map.stem) for _map in tqdm(maps, f'processing maps')
                     # some maps seem to be duplicates with (1) in name
                     if '(1)' not in _map.stem]
    logging.info(f'remaining {len(maps)} maps after clean up')

    # kialo domains
    main_domains = []
    if args['training_domain_index'] >= 0:
        maps2uniquetopic, (_, _, main2subtopic) = get_maps2uniquetopic('data/kialomaps2maintopics.tsv',
                                                                       'data/kialo_domains.tsv')
        main_domains = list(main2subtopic.keys())

        # domain_argument_maps = {domain: [KialoMap(str(data_path / (map_name + '.txt')), map_name)
        #                                  for map_name, map_domain in maps2uniquetopic.items() if map_domain == domain]
        #                         for domain in main2subtopic}
        domain_argument_maps = {domain: [] for domain in main2subtopic}
        for argument_map in argument_maps:
            if argument_map.name in maps2uniquetopic:
                domain_argument_maps[maps2uniquetopic[argument_map.name]].append(argument_map)
            else:
                logging.warning(argument_map.name, ' skipped!')
        print(f'{len(domain_argument_maps)=}')
        argument_maps = domain_argument_maps[main_domains[args['training_domain_index']]]
        args['training_domain'] = main_domains[args['training_domain_index']]

    # split data
    argument_maps_train, argument_maps_test = train_test_split(argument_maps, test_size=0.2, random_state=42)
    logging.info(f'train/eval using {len(argument_maps_train)=} {len(argument_maps_test)=}')

    model_save_path = get_model_save_path(model_name, args)
    logging.info(f'{model_save_path=}')
    logging.getLogger().handlers[0].flush()

    if args['local']:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project='argument-maps', name=model_save_path,
               # to fix "Error communicating with wandb process"
               # see https://docs.wandb.ai/guides/track/launch#init-start-error
               settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)

    if args['do_train']:

        # prepare samples
        maps_samples = {x.label: [] for x in argument_maps_train}
        # maps_samples_dev = {x.label: [] for x in argument_maps}
        for i, argument_map in enumerate(argument_maps_train):
            argument_map_util = Evaluation(argument_map, no_ranks=True)
            for child, parent in zip(argument_map_util.child_nodes, argument_map_util.parent_nodes):
                if args['hard_negatives']:
                    non_parents = [x for x in argument_map_util.parent_nodes if x != parent]
                    if len(non_parents) > args['hard_negatives_size'] > 0:
                        non_parents = random.sample(non_parents, args['hard_negatives_size'])
                    for non_parent in non_parents:
                        # NOTE original code also adds opposite
                        maps_samples[argument_map.label].append(
                            InputExample(texts=[x.name for x in [child, parent, non_parent]]))
                else:
                    maps_samples[argument_map.label].append(
                        InputExample(texts=[x.name for x in [child, parent]]))
        if args['debug_size']:
            maps_samples = {k: x[:args['debug_size']] for k, x in maps_samples.items()}

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_samples = list(itertools.chain(*maps_samples.values()))

        logging.info("Train samples: {}".format(len(train_samples)))

        # Special data loader that avoid duplicates within a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        print(f'{len(train_dataloader)=}')
        # 10% of train data for warm-up
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
        logging.info("Warmup-steps: {}".format(warmup_steps))

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path,
                  use_amp=False  # Set to True, if your GPU supports FP16 operations
                  )
    # eval
    if args['do_eval']:
        eval(model_save_path, args, argument_maps_test,
             domain=main_domains[args['training_domain_index']] if args['training_domain_index'] >= 0 else 'all')
        if args['training_domain_index'] >= 0:
            for domain in main_domains[:args['training_domain_index']] + main_domains[args['training_domain_index']+1:]:
                eval(model_save_path, args, argument_maps_test, domain=domain)


def eval(output_dir, args, argument_maps, domain):
    model = SentenceTransformer(args['eval_model_name_or_path'] if args['eval_model_name_or_path'] else
                                output_dir)
    results_path = Path(output_dir + '-results') / domain
    results_path.mkdir(exist_ok=True, parents=True)
    encoder_mulitlingual = MapEncoder(max_seq_len=128,
                                      sbert_model_identifier=None,
                                      model=model,
                                      normalize_embeddings=True)
    all_results = []
    maps_all_results = {}
    try:
        for j, eval_argument_map in enumerate(argument_maps):
            results = evaluate_map(encoder_mulitlingual, eval_argument_map, {"Pro", "Con"})
            maps_all_results[eval_argument_map.label] = results
            all_results.append(results)
    except Exception as e:
        (results_path / f'all_maps.json').write_text(json.dumps(maps_all_results))
        wandb.log({'test': maps_all_results})
        wandb.log({'test': all_results})
        data = [[map_name.rsplit('-', 1)[-1], v] for map_name, v in maps_all_results.items()]
        table = wandb.Table(data=data, columns=["map id", "scores"])
        wandb.log({'test': {'detailed': wandb.plot.line(
            table, "map id", "score", title="Detailed results per map id")}})
        raise e

    avg_results = {
        key: {inner_key: sum(entry[key][inner_key] for entry in all_results) / len(all_results) for inner_key in value}
        for key, value in all_results[0].items()}
    (results_path / f'-avg.json').write_text(json.dumps(avg_results))
    wandb.log({'test': {domain: {'avg': avg_results}}})


if __name__ == '__main__':
    main()
