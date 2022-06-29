import sys
import train_triplets_kialo


sys.argv += ['--local', 'True']
sys.argv += ['--debug_maps_size', '10']
# sys.argv += '--eval_model_name_or_path sentence-transformers/paraphrase-multilingual-mpnet-base-v2'.split(' ')
sys.argv += '--eval_model_name_or_path sentence-transformers/paraphrase-albert-small-v2'.split(' ')
# sys.argv += '--eval_not_trained True --do_train False'.split(' ')

# debug specific map
# sys.argv += '--no_data_split True --debug_map_index 22187'.split(' ')

# philosophy domain
sys.argv += '--training_domain_index 15 --debug_maps_size 1000'.split(' ')

train_triplets_kialo.main()
