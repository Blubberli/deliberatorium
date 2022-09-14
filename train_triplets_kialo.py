import faulthandler
import itertools
import json
import logging
import math
import os
import random
import signal
import statistics
from pathlib import Path

import wandb
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample, CrossEncoder
from sentence_transformers import models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util import semantic_search
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import set_seed

import templates
from argumentMap import KialoMap
from childNode import ChildNode
from eval_util import METRICS, evaluate_map, format_metrics
from encode_nodes import MapEncoder
from evaluation import Evaluation
from kialo_domains_util import get_maps2uniquetopic
from kialo_util import read_data, read_annotated_maps_ids, read_annotated_samples
from templates import format
from util import remove_url_and_hashtags
from train_triplets_delib import parse_args, get_model_save_path

AVAILABLE_MAPS = ['dopariam1', 'dopariam2', 'biofuels', 'RCOM', 'CI4CG']

logging.basicConfig(format='%(asctime)s,%(msecs)d p%(process)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def add_more_args(parser):
    parser.add_argument('--debug_map_index', type=str, default=None)
    parser.add_argument('--no_data_split', type=str, default=None)
    parser.add_argument('--training_domain_index', type=int, default=-1)
    parser.add_argument('--use_templates', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--annotated_samples_in_test', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_dev', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--max_candidates', type=int, default=0)
    parser.add_argument('--do_eval_annotated_samples', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--rerank', type=lambda x: (str(x).lower() == 'true'), default=False)


def main():
    faulthandler.register(signal.SIGUSR1.value)

    args = parse_args(add_more_args)

    seed = 42
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    model_name = args['model_name_or_path']
    train_batch_size = args['train_batch_size']  # The larger you select this, the better the results (usually)
    max_seq_length = args['max_seq_length']
    num_epochs = args['num_train_epochs']

    model_save_path = get_model_save_path(model_name, args)
    logging.info(f'{model_save_path=}')
    logging.getLogger().handlers[0].flush()

    if args['local']:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project='argument-maps', name=model_save_path,
               # to fix "Error communicating with wandb process"
               # see https://docs.wandb.ai/guides/track/launch#init-start-error
               settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args | {'data': 'kialoV2'})

    data_splits = None
    main_domains = []
    if args['do_train'] or args['do_eval']:
        argument_maps = read_data(args)

        if args['training_domain_index'] >= 0:
            maps2uniquetopic, (_, _, main2subtopic) = get_maps2uniquetopic('data/kialoID2MainTopic.csv',
                                                                           'data/kialo_domains.tsv')
            main_domains = list(main2subtopic.keys())

            # domain_argument_maps = {domain: [KialoMap(str(data_path / (map_name + '.txt')), map_name)
            #                                  for map_name, map_domain in maps2uniquetopic.items() if map_domain == domain]
            #                         for domain in main2subtopic}
            domain_argument_maps = {domain: [] for domain in main2subtopic}
            for argument_map in argument_maps:
                if argument_map.id in maps2uniquetopic:
                    domain_argument_maps[maps2uniquetopic[argument_map.id]].append(argument_map)
                else:
                    logging.warning(f'{argument_map.label} {argument_map.name} skipped!')
            argument_maps = domain_argument_maps[main_domains[args['training_domain_index']]]
            args['training_domain'] = main_domains[args['training_domain_index']]
            logging.info(f"{args['training_domain']=}")
            logging.info(f"{len(argument_maps)=} maps in domain args['training_domain_index']={args['training_domain']}")
            wandb.config.update(args | {'data': 'kialoV2'})

        data_splits = split_data(argument_maps, args, model_save_path, seed)

    if args['do_train']:

        maps_samples = prepare_samples(data_splits['train'], 'train', args)
        maps_samples_dev = prepare_samples(data_splits['dev'], 'dev', args)

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_samples = list(itertools.chain(*maps_samples.values()))
        dev_samples = list(itertools.chain(*maps_samples_dev.values()))

        logging.info("Train samples: {}".format(len(train_samples)))

        # Special data loader that avoid duplicates within a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        dev_evaluator = (EmbeddingSimilarityEvaluator.from_input_examples(
            dev_samples, batch_size=train_batch_size, name='dev')
                         if args['use_dev'] else None)

        print(f'{len(train_dataloader)=}')
        # 10% of train data for warm-up
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
        logging.info("Warmup-steps: {}".format(warmup_steps))

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=num_epochs,
                  evaluator=dev_evaluator,
                  evaluation_steps=args['eval_steps'] if args['eval_steps'] else
                  (int(len(train_dataloader) * 0.1) if dev_evaluator else 0),
                  warmup_steps=warmup_steps,
                  output_path=model_save_path,
                  use_amp=False  # Set to True, if your GPU supports FP16 operations
                  )

    # eval
    if args['do_eval'] or args['do_eval_annotated_samples']:
        model = SentenceTransformer(args['eval_model_name_or_path'] if args['eval_model_name_or_path'] else
                                    model_save_path)
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') if args['rerank'] else None
        if args['do_eval']:
            map_encoder = MapEncoder(max_seq_len=args['max_seq_length'],
                                     sbert_model_identifier=None,
                                     model=model,
                                     normalize_embeddings=True,
                                     use_templates=args['use_templates'])
            all_results = []
            all_results.extend(
                eval(model_save_path, data_splits['test'],
                     domain=main_domains[args['training_domain_index']] if args['training_domain_index'] >= 0 else 'all',
                     max_candidates=args['max_candidates'],
                     map_encoder=map_encoder, cross_encoder=cross_encoder))
            if args['training_domain_index'] >= 0:
                for domain in main_domains[:args['training_domain_index']] + main_domains[args['training_domain_index']+1:]:
                    all_results.extend(eval(model_save_path, domain_argument_maps[domain], domain=domain,
                                            max_candidates=args['max_candidates'],
                                            map_encoder=map_encoder, cross_encoder=cross_encoder))
            avg_results = get_avg(all_results)
            (Path(model_save_path + '-results') / f'-avg.json').write_text(json.dumps(avg_results))
            wandb.log({'test': {'avg': avg_results}})

        if args['do_eval_annotated_samples']:
            eval_samples(model_save_path, args, model, cross_encoder)


def split_data(argument_maps: list[KialoMap], args: dict, output_dir: str, seed: int):
    test_size = 0.2
    data_splits = {}
    annotated_maps, remaining_maps = [], argument_maps
    if args['annotated_samples_in_test']:
        annotated_maps_ids = read_annotated_maps_ids(args['local'])
        not_annotated_maps, annotated_maps = [], []
        for argument_map in argument_maps:
            (not_annotated_maps, annotated_maps)[argument_map.id in annotated_maps_ids].append(argument_map)

        more_test_maps_num = round(test_size * len(argument_maps) - len(annotated_maps))
        logging.info(f'keep annotated samples in test: {len(annotated_maps)=} + {more_test_maps_num} for {test_size=}')
        # change test_size from percentage to absolute number of maps to include beside the annotated samples
        test_size = more_test_maps_num
        remaining_maps = not_annotated_maps

    data_splits['train'], data_splits['test'] = (
        train_test_split(remaining_maps, test_size=test_size, random_state=seed)
        if not args['no_data_split'] else (remaining_maps, remaining_maps))
    data_splits['test'] = annotated_maps + data_splits['test']

    if args['use_dev']:
        data_splits['train'], data_splits['dev'] = train_test_split(data_splits['train'],
                                                                    test_size=0.2, random_state=seed)
    else:
        data_splits['dev'] = []
    logging.info('train/dev/test using sizes: ' + ' '.join([f'{k}={len(v)} ({(len(v) / len(argument_maps)):.2f})'
                                                            for k, v in data_splits.items()]))

    # save split ids
    path = Path(output_dir + '-data')
    path.mkdir(exist_ok=True, parents=True)
    for split_name, split in data_splits.items():
        (path/f'{split_name}.json').write_text(json.dumps([x.id for x in split]))

    return data_splits


def prepare_samples(argument_maps, split, args):
    maps_samples = {x.label: [] for x in argument_maps}
    for i, argument_map in enumerate(tqdm(argument_maps, f'preparing samples {split}')):
        argument_map_util = Evaluation(argument_map, no_ranks=True, max_candidates=args['max_candidates'])
        for child, parent in zip(argument_map_util.child_nodes, argument_map_util.parent_nodes):
            if split == 'dev' or args['hard_negatives']:
                non_parents = [x for x in argument_map_util.parent_nodes if x != parent]
                if len(non_parents) > args['hard_negatives_size'] > 0:
                    non_parents = random.sample(non_parents, args['hard_negatives_size'])

                if split == 'dev':
                    maps_samples[argument_map.label].extend([create_training_example(
                        [child, non_parent], label=0) for non_parent in non_parents])
                    maps_samples[argument_map.label].append(create_training_example([child, parent], label=1))
                else:
                    # NOTE original code also adds opposite
                    maps_samples[argument_map.label].extend([create_training_example(
                        [child, parent, non_parent], args['use_templates']) for non_parent in non_parents])
            else:
                maps_samples[argument_map.label].append(create_training_example(
                    [child, parent], args['use_templates']))
    if args['debug_size']:
        maps_samples = {k: x[:args['debug_size']] for k, x in maps_samples.items()}
    return maps_samples


def create_training_example(nodes: list[ChildNode], use_templates=False, label=0):
    types = ['child'] + ['parent'] * (len(nodes) - 1)
    return InputExample(texts=[templates.format(x.name, t, use_templates) for x, t in zip(nodes, types)], label=label)


def eval(output_dir, argument_maps, domain, max_candidates, map_encoder: MapEncoder, cross_encoder: CrossEncoder):
    results_path = Path(output_dir + '-results') / domain
    results_path.mkdir(exist_ok=True, parents=True)
    all_results = []
    maps_all_results = {}
    nodes_all_results = {}
    try:
        for j, eval_argument_map in enumerate(tqdm(argument_maps, f'eval maps in domain {domain}')):
            try:
                results, nodes_all_results[eval_argument_map.label] = evaluate_map(map_encoder,
                                                                                   eval_argument_map, {1, -1},
                                                                                   max_candidates=max_candidates,
                                                                                   cross_encoder=cross_encoder)
            except Exception as e:
                logging.error('cannot evaluate map ' + eval_argument_map.label)
                raise e
            maps_all_results[eval_argument_map.label] = results
            all_results.append(results)
    finally:
        (results_path / f'all_maps.json').write_text(json.dumps(maps_all_results))
        (results_path / f'all_nodes.json').write_text(json.dumps(nodes_all_results))
        # wandb.log({'test': maps_all_results})
        # wandb.log({'test': all_results})
        # data = [[map_name.rsplit('-', 1)[-1], v] for map_name, v in maps_all_results.items()]
        # table = wandb.Table(data=data, columns=["map id", "scores"])
        # wandb.log({'test': {'detailed': wandb.plot.line(
        #     table, "map id", "score", title="Detailed results per map id")}})

    avg_results = get_avg(all_results)
    (results_path / f'-avg.json').write_text(json.dumps(avg_results))
    wandb.log({'test': {domain: {'avg': avg_results}}})
    return all_results


def eval_samples(output_dir, args, encoder: SentenceTransformer, cross_encoder: CrossEncoder):
    results_path = Path(output_dir + '-results')
    results_path.mkdir(exist_ok=True, parents=True)

    samples = read_annotated_samples(args['local'], args)
    for node_id, sample in tqdm(samples.items(), desc='encode and evaluate annotated samples'):
        sample['text'] = remove_url_and_hashtags(sample['text'])
        candidates = []
        for candidate_id, candidate in sample['candidates'].items():
            candidates.append({'text': remove_url_and_hashtags(candidate['text']), 'id': candidate_id})

        node_embedding = encoder.encode(format(sample['text'], 'child', args['use_templates']), convert_to_tensor=True,
                                        show_progress_bar=False)
        candidates_embedding = encoder.encode([format(x['text'], 'parent', args['use_templates']) for x in candidates],
                                              convert_to_tensor=True,
                                              show_progress_bar=False)
        hits = semantic_search(node_embedding, candidates_embedding, top_k=len(sample['candidates']))[0]
        predictions = []
        rank = -1
        for i, hit in enumerate(hits, start=1):
            candidate = candidates[hit['corpus_id']]
            candidate['score'] = hit['score']
            predictions.append(candidate)
            if candidate['id'] == sample['parent ID']:
                rank = i
        sample['predictions'] = predictions
        sample['rank'] = rank
    metrics = Evaluation.calculate_metrics([x['rank'] for x in samples.values()])
    logging.info(format_metrics(metrics))
    (results_path / 'annotated_samples_predictions.json').write_text(json.dumps(samples))
    (results_path / 'annotated_samples_metrics.json').write_text(json.dumps(metrics))


def get_avg(all_results):
    avg_results = {
        key: {inner_key: statistics.fmean(entry[key][inner_key] for entry in all_results if entry[key])
              for inner_key in METRICS}
        for key, value in all_results[0].items()}
    return avg_results


if __name__ == '__main__':
    main()
