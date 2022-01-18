import argparse
import itertools
import logging
import math
from datetime import datetime
from pathlib import Path
from pprint import pprint

from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers import models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from argumentMap import ArgumentMap
from baseline import evaluate_map
from encode_nodes import MapEncoder
from evaluation import Evaluation

AVAILABLE_MAPS = ['doppariam1', 'dopariam2', 'biofuels', 'RCOM', 'CI4CG']

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--do_train', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--do_eval', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--model_name_or_path', help="model", type=str, default='xlm-roberta-base')
    parser.add_argument('--eval_model_name_or_path', help="model", type=str, default=None)
    parser.add_argument('--lang', help="english, italian, *", type=str, default='*')
    parser.add_argument('--argument_map', help="argument map from (doppariam1, doppariam2, biofuels, RCOM, CI4CG)",
                        type=str, default='*')
    parser.add_argument('--hard_negatives', type=lambda x: (str(x).lower() == 'true'), default=True)
    args = vars(parser.parse_args())
    assert args['argument_map'] in AVAILABLE_MAPS, \
        f"{args['argument_map']=} is not a value from (doppariam1, doppariam2, biofuels, RCOM, CI4CG)"
    pprint(args)
    return args


def main():
    args = parse_args()
    model_name = args['model_name_or_path']
    train_batch_size = 128  # The larger you select this, the better the results (usually)
    max_seq_length = 75
    num_epochs = 1

    model_save_path_prefix = 'model_' + model_name.replace("/", "-")

    data_path = (Path.home() / "data/e-delib/deliberatorium/maps" if args['local'] else
                 Path("/mount/projekte/e-delib/data/deliberatorium/maps"))
    maps = list(data_path.glob(f"{args['lang']}_maps/{args['argument_map']}.json"))
    logging.info('processing maps: ', maps)
    argument_maps = [ArgumentMap(str(_map)) for _map in maps]
    maps_samples = [[]] * len(argument_maps)
    print(len(argument_maps))
    for i, argument_map in enumerate(argument_maps):
        argument_map_util = Evaluation(argument_map, no_ranks=True)
        for child, parent in zip(argument_map_util.child_nodes, argument_map_util.parent_nodes):
            if args['hard_negatives']:
                for non_parent in [x for x in argument_map_util.parent_nodes if x != parent]:
                    # NOTE original code also adds opposite
                    maps_samples[i].append(InputExample(texts=[x._name for x in [child, parent, non_parent]]))
            else:
                maps_samples[i].append(InputExample(texts=[x._name for x in [child, parent]]))

    for i, argument_map in enumerate(argument_maps):
        if args['do_train']:
            word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            train_samples = list(itertools.chain(*(maps_samples[:i] + maps_samples[i + 1:])))
            dev_samples = maps_samples[i]

            logging.info("Training using: {}".format([x._name for x in argument_maps[:i] + argument_maps[i + 1:]]))
            logging.info("Evaluating using: {}".format(argument_map._name))
            logging.info("Train samples: {}".format(len(train_samples)))
            logging.info("Dev samples: {}".format(len(dev_samples)))

            # Special data loader that avoid duplicates within a batch
            train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model)

            dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                             name='sts-dev')

            # 10% of train data for warm-up
            warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
            logging.info("Warmup-steps: {}".format(warmup_steps))

            model_save_path = model_save_path_prefix + f'-{argument_map._name}-' + datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S")
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      epochs=num_epochs,
                      # no dev for now
                      # evaluator=dev_evaluator,
                      # evaluation_steps=int(len(train_dataloader) * 0.1),
                      warmup_steps=warmup_steps,
                      output_path=model_save_path,
                      use_amp=False  # Set to True, if your GPU supports FP16 operations
                      )

        if args['do_eval']:
            model = SentenceTransformer(
                args['eval_model_name_or_path'] if args['eval_model_name_or_path'] else model_save_path)
            encoder_mulitlingual = MapEncoder(max_seq_len=128,
                                              sbert_model_identifier=None,
                                              model=model,
                                              normalize_embeddings=True, use_descriptions=False)
            evaluate_map(encoder_mulitlingual, argument_map)


if __name__ == '__main__':
    main()
