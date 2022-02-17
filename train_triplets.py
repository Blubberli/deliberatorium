import argparse
import itertools
import json
import logging
import math
from pathlib import Path
from pprint import pprint

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--debug_size', type=int, default=0)
    parser.add_argument('--do_train', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--do_eval', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--model_name_or_path', help="model", type=str, default='xlm-roberta-base')
    parser.add_argument('--eval_model_name_or_path', help="model", type=str, default=None)
    parser.add_argument('--output_dir_label', type=str)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--lang', help="english, italian, *", type=str, default='*')
    parser.add_argument('--argument_map',
                        help=f"argument map from {', '.join(AVAILABLE_MAPS)} to train on",
                        type=str, default=None)
    parser.add_argument('--argument_map_dev',
                        help=f"argument map from {', '.join(AVAILABLE_MAPS)} to use as dev",
                        type=str, default=None)
    parser.add_argument('--train_on_one_map',
                        help="either train on `argument_map` and eval on all others or train on all others and evaluate on `argument_map`",
                        type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--hard_negatives', type=lambda x: (str(x).lower() == 'true'), default=True)
    args = vars(parser.parse_args())
    assert (not args['argument_map'] or args['argument_map'] in AVAILABLE_MAPS), \
        f"{args['argument_map']=} is not a value from: {', '.join(AVAILABLE_MAPS)}"
    pprint(args)
    return args


def get_model_save_path(model_name, map_label, map_label_dev, train_on_one_map, output_dir_label):
    model_save_path_prefix = 'results/' + model_name.replace("/", "-")
    return model_save_path_prefix + \
        (f'-{output_dir_label}' if output_dir_label else '') + \
        ('-trained' if train_on_one_map else '-evaluated') + f'-on-{map_label}' + \
        (f'-dev-{map_label_dev}' if map_label_dev else '')


def main():
    args = parse_args()

    if args['debug_size'] > 0:
        logging.info(f"!!!!!!!!!!!!!!! DEBUGGING with {args['debug_size']}")
    
    if args['argument_map'] and args['argument_map'] == args['argument_map_dev']:
        logging.info('same value for argument_map and argument_map_dev! exiting')
        exit()
    
    model_name = args['model_name_or_path']
    train_batch_size = 128  # The larger you select this, the better the results (usually)
    max_seq_length = 75
    num_epochs = args['num_train_epochs']

    data_path = (Path.home() / "data/e-delib/deliberatorium/maps" if args['local'] else
                 Path("/mount/projekte/e-delib/data/deliberatorium/maps"))
    maps = list(data_path.glob(f"{args['lang']}_maps/*.json"))
    logging.info(f'processing {len(maps)} maps: ' + str(maps))
    argument_maps = [DeliberatoriumMap(str(_map), _map.stem) for _map in maps]

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
        model_save_path = get_model_save_path(model_name, argument_map_label, args['argument_map_dev'],
                                              args['train_on_one_map'],
                                              args['output_dir_label'])
        logging.info(f'{model_save_path=}')
        logging.getLogger().handlers[0].flush()

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

            dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                             name=args['argument_map_dev'])

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
            model = SentenceTransformer(args['eval_model_name_or_path'] if args['eval_model_name_or_path'] else
                                        model_save_path)
            eval_argument_maps = ((argument_maps[:i] + argument_maps[i + 1:]) if args['train_on_one_map'] else
                                  [argument_maps[i]])

            results_path = Path(model_save_path + '-results')
            results_path.mkdir(exist_ok=True)

            for eval_argument_map in eval_argument_maps:
                encoder_mulitlingual = MapEncoder(max_seq_len=128,
                                                  sbert_model_identifier=None,
                                                  model=model,
                                                  normalize_embeddings=True, use_descriptions=False)
                results = evaluate_map(encoder_mulitlingual, eval_argument_map, {"issue", "idea"})
                (results_path / f'{eval_argument_map.label}.json').write_text(json.dumps(results))


if __name__ == '__main__':
    main()
