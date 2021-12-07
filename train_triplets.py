import itertools
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers import models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from argumentMap import ArgumentMap
from baseline import evaluate_map
from encode_nodes import MapEncoder
from evaluation import Evaluation

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model_name = sys.argv[1] if len(sys.argv) > 1 else 'xlm-roberta-base'
train_batch_size = 128  # The larger you select this, the better the results (usually)
max_seq_length = 75
num_epochs = 1

model_save_path_prefix = 'model_' + model_name.replace("/", "-")

data_path = Path.home() / "data/e-delib/deliberatorium/maps/italian_maps"
data_path = Path.home() / "/mount/projekte/e-delib/data/deliberatorium/maps/italian_maps/"
maps = os.listdir(data_path)
argument_maps = [ArgumentMap("%s/%s" % (str(data_path), _map)) for _map in maps]
maps_samples = [[]] * len(maps)
print(len(maps))
for i, argument_map in enumerate(argument_maps):
    argument_map_util = Evaluation(argument_map, no_ranks=True)
    for child, parent in zip(argument_map_util.child_nodes, argument_map_util.parent_nodes):
        for non_parent in [x for x in argument_map_util.parent_nodes if x != parent]:
            # NOTE original code also adds opposite
            maps_samples[i].append(InputExample(texts=[x._name for x in [child, parent, non_parent]]))

for i, argument_map in enumerate(argument_maps):
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

    model_save_path = model_save_path_prefix + f'-{maps[i]}-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              # no dev for now
              # evaluator=dev_evaluator,
              # evaluation_steps=int(len(train_dataloader) * 0.1),
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              use_amp=False  # Set to True, if your GPU supports FP16 operations
              )

    model = SentenceTransformer(model_save_path)
    encoder_mulitlingual = MapEncoder(max_seq_len=128,
                                      sbert_model_identifier=None,
                                      model=model,
                                      normalize_embeddings=True, use_descriptions=False)
    encoder_mulitlingual.encode_argument_map(argument_map)
    evaluate_map(encoder_mulitlingual, argument_map)
