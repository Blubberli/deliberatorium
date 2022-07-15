import argparse
import faulthandler
import itertools
import json
import logging
import math
import signal
from pathlib import Path
from pprint import pprint

import wandb
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers import models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from argumentMap import DeliberatoriumMap
from baseline import evaluate_map
from encode_nodes import MapEncoder
from evaluation import Evaluation

AVAILABLE_MAPS = ['dopariam1', 'dopariam2', 'biofuels', 'RCOM', 'CI4CG']

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args(add_more_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--debug_size', type=int, default=0)
    parser.add_argument('--debug_maps_size', type=int, default=0)
    parser.add_argument('--do_train', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--do_eval', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--eval_not_trained', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--model_name_or_path', help="model", type=str)
    parser.add_argument('--eval_model_name_or_path', help="model", type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default=None)
    parser.add_argument('--output_dir_label', type=str)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--lang', help="english, italian, *", type=str, default='*')
    parser.add_argument('--use_descriptions', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--argument_map',
                        help=f"argument map from {', '.join(AVAILABLE_MAPS)} to train on",
                        type=str, default=None)
    parser.add_argument('--argument_map_dev',
                        help=f"argument map from {', '.join(AVAILABLE_MAPS)} to use as dev",
                        type=str, default=None)
    parser.add_argument('--train_on_one_map',
                        help="either train on `argument_map` and eval on all others or train on all others and evaluate on `argument_map`",
                        type=lambda x: (str(x).lower() == 'true'), default=False)
    if add_more_args:
        add_more_args(parser)
    parser.add_argument('--hard_negatives', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--hard_negatives_size', type=int, default=-1)
    args = vars(parser.parse_args())
    assert (not args['argument_map'] or args['argument_map'] in AVAILABLE_MAPS), \
        f"{args['argument_map']=} is not a value from: {', '.join(AVAILABLE_MAPS)}"
    pprint(args)

    if args['debug_size'] > 0:
        logging.info(f"!!!!!!!!!!!!!!! DEBUGGING with {args['debug_size']} samples")

    if args['debug_maps_size'] > 0:
        logging.info(f"!!!!!!!!!!!!!!! DEBUGGING with {args['debug_maps_size']} maps")

    if args['argument_map'] and args['argument_map'] == args['argument_map_dev']:
        logging.info('same value for argument_map and argument_map_dev! exiting')
        exit()

    return args


def get_model_save_path(model_name, args, map_label=None):
    model_save_path_prefix = 'results/' + (f'{args["output_dir_prefix"]}/' if args['output_dir_prefix'] else '')\
        + (f"domain{args['training_domain_index']}"
           if 'training_domain_index' in args and args['training_domain_index'] >= 0 else '')
                             # + model_name.replace("/", "-")
    if not map_label:
        return model_save_path_prefix
    return model_save_path_prefix + \
        (f'-{args["output_dir_label"]}' if args['output_dir_label'] else '') + \
        ('-trained' if args['train_on_one_map'] else '-evaluated') + f'-on-{map_label}' + \
        (f'-dev-{args["argument_map_dev"]}' if args['argument_map_dev'] else '')


def main():
    faulthandler.register(signal.SIGUSR1.value)

    args = parse_args()
    
    model_name = args['model_name_or_path']
    train_batch_size = args['train_batch_size']  # The larger you select this, the better the results (usually)
    max_seq_length = 75
    num_epochs = args['num_train_epochs']

    data_path = (Path.home() / "data/e-delib/deliberatorium/maps" if args['local'] else
                 Path("/mount/projekte/e-delib/data/deliberatorium/maps"))
    maps = list(data_path.glob(f"{args['lang']}_maps/*.json"))
    logging.info(f'processing {len(maps)} maps: ' + str(maps))
    argument_maps = [DeliberatoriumMap(str(_map), _map.stem) for _map in maps]
    
    if args['eval_not_trained']:
        logging.info('eval all arguments as not part of training data')
        save_path = get_model_save_path(args['eval_model_name_or_path'], args)
        wandb.init(project='argument-maps', name=save_path,
                   # to fix "Error communicating with wandb process"
                   # see https://docs.wandb.ai/guides/track/launch#init-start-error
                   settings=wandb.Settings(start_method="fork"))
        wandb.config.update(args)
        eval(save_path, args, argument_maps)
        exit()

    # prepare samples
    maps_samples = {x.label: [] for x in argument_maps}
    maps_samples_dev = {x.label: [] for x in argument_maps}
    for i, argument_map in enumerate(argument_maps):
        argument_map_util = Evaluation(argument_map, no_ranks=True)
        for child, parent in zip(argument_map_util.child_nodes, argument_map_util.parent_nodes):
            for non_parent in [x for x in argument_map_util.parent_nodes if x != parent]:
                if args['hard_negatives']:
                    # NOTE original code also adds opposite
                    maps_samples[argument_map.label].append(
                        InputExample(texts=[x.name for x in [child, parent, non_parent]]))
                maps_samples_dev[argument_map.label].append(
                    InputExample(texts=[x.name for x in [child, non_parent]], label=0))
            if not args['hard_negatives']:
                maps_samples[argument_map.label].append(
                    InputExample(texts=[x.name for x in [child, parent]]))
            maps_samples_dev[argument_map.label].append(
                    InputExample(texts=[x.name for x in [child, parent]], label=1))
    if args['debug_size']:
        maps_samples = {k: x[:args['debug_size']] for k, x in maps_samples.items()}
        maps_samples_dev = {k: x[:(args['debug_size']//5)] for k, x in maps_samples_dev.items()}

    # for each map: train/eval
    for i, argument_map_label in enumerate(maps_samples.keys()):
        if args['argument_map'] and args['argument_map'] not in str(maps[i]):
            continue
        model_save_path = get_model_save_path(model_name, args, argument_map_label)
        logging.info(f'{model_save_path=}')
        logging.getLogger().handlers[0].flush()

        wandb.init(project='argument-maps', name=model_save_path,
                   # to fix "Error communicating with wandb process"
                   # see https://docs.wandb.ai/guides/track/launch#init-start-error
                   settings=wandb.Settings(start_method="fork"))
        wandb.config.update(args)

        if args['do_train']:
            word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            if args['train_on_one_map']:
                train_samples = maps_samples[argument_map_label]
            else:
                train_maps = [x for k, x in maps_samples.items()
                              if k != argument_map_label and k != args['argument_map_dev']]
                train_samples = list(itertools.chain(*train_maps))
            dev_samples = next((x for k, x in maps_samples_dev.items()
                                if k == args['argument_map_dev']), [])

            logging.info("Training using: {}".format([x.name for x in argument_maps[:i] + argument_maps[i + 1:]]))
            logging.info("Evaluating using: {}".format(argument_map_label))
            logging.info("Train samples: {}".format(len(train_samples)))
            logging.info("Dev samples: {}".format(len(dev_samples)))

            # Special data loader that avoid duplicates within a batch
            train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model)

            dev_evaluator = (EmbeddingSimilarityEvaluator.from_input_examples(
                dev_samples, batch_size=train_batch_size, name=args['argument_map_dev'])
                             if args['argument_map_dev'] else None)

            print(f'{len(train_dataloader)=}')
            # 10% of train data for warm-up
            warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
            logging.info("Warmup-steps: {}".format(warmup_steps))

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      epochs=num_epochs,
                      evaluator=dev_evaluator,
                      evaluation_steps=args['eval_steps'] if args['eval_steps'] else
                      (int(len(train_dataloader) * 0.1) if args['argument_map_dev'] else 0),
                      warmup_steps=warmup_steps,
                      output_path=model_save_path,
                      use_amp=False  # Set to True, if your GPU supports FP16 operations
                      )
        # eval
        if args['do_eval']:
            eval(model_save_path, args, argument_maps, i)


def eval(output_dir, args, argument_maps, training_map_index=-1):
    model = SentenceTransformer(args['eval_model_name_or_path'] if args['eval_model_name_or_path'] else
                                output_dir)
    results_path = Path(output_dir + '-results')
    results_path.mkdir(exist_ok=True, parents=True)
    encoder_mulitlingual = MapEncoder(max_seq_len=128,
                                      sbert_model_identifier=None,
                                      model=model,
                                      normalize_embeddings=True, use_descriptions=args['use_descriptions'])
    for j, eval_argument_map in enumerate(argument_maps):
        train_eval = ((args['train_on_one_map'] and training_map_index == j) or
                      (not args['train_on_one_map'] and training_map_index != j)
                      and training_map_index >= 0)
        results, _ = evaluate_map(encoder_mulitlingual, eval_argument_map, {"issue", "idea"})
        (results_path / f'{eval_argument_map.label}{"-train" if train_eval else ""}.json'). \
            write_text(json.dumps(results))
        wandb.log({eval_argument_map.label: results})
        wandb.log(results)
        # TODO add dev
        split = 'train' if train_eval else 'test'
        wandb.log({split: {eval_argument_map.label: results}})
        wandb.log({split: results})


if __name__ == '__main__':
    main()
