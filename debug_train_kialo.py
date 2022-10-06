import sys
import train_triplets_kialo


sys.argv += ['--local', 'True']
sys.argv += ['--debug_maps_size', '10']

sys.argv += '--model_name_or_path sentence-transformers/paraphrase-albert-small-v2'.split(' ')
sys.argv += '--hard_negatives False'.split(' ')
sys.argv += '--lang english'.split(' ')
sys.argv += '--use_templates True'.split(' ')
# sys.argv += '--template_id pro/con'.split(' ')
# sys.argv += '--use_dev True'.split(' ')


sys.argv += '--data_samples_seed 13'.split(' ')
sys.argv += '--debug_maps_size 13 --train_maps_size 1 --train_per_map_size 8 --train_batch_size 8'.split(' ')

# sys.argv += '--train_method class'.split(' ')
sys.argv += '--train_method cossco'.split(' ')

# sys.argv += '--eval_model_name_or_path sentence-transformers/paraphrase-albert-small-v2'.split(' ')
# sys.argv += '--eval_not_trained True --do_train False'.split(' ')

# sys.argv += '--rerank True'.split(' ')

sys.argv += '--annotated_samples_in_test True --debug_map_index 17763'.split(' ')

# eval annotated samples
# sys.argv += '--do_eval False --do_eval_annotated_samples True --debug_maps_size 3'.split(' ')

# debug specific map
# sys.argv += '--no_data_split True --debug_map_index 22187'.split(' ')

# philosophy domain
# sys.argv += '--training_domain_index 16 --debug_maps_size 1000'.split(' ')

train_triplets_kialo.main()
