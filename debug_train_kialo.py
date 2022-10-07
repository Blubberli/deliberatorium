import sys
import train_triplets_kialo

# sys.argv += '--train_method cossco --template_id None --train_maps_size 1 --train_per_map_size 8 --data_samples_seed 100 --train_batch_size 16 --lr 0.001'.split(' ')
# sys.argv += '--lang english --model_name_or_path sentence-transformers/all-mpnet-base-v2 --hard_negatives False --annotated_samples_in_test True --do_eval_annotated_samples True --model_name_or_path sentence-transformers/all-mpnet-base-v2 --train_method mulneg --template_id None --train_maps_size 1 --train_per_map_size 8 --data_samples_seed 13 --train_batch_size 8 --output_dir_prefix results/few-shot-seeds-mulneg/all-mpnet-mulneg-n8in1-seed13'.split(' ')
# sys.argv += '--lang english --model_name_or_path sentence-transformers/all-mpnet-base-v2 --hard_negatives False --annotated_samples_in_test True --do_eval_annotated_samples True --model_name_or_path sentence-transformers/all-mpnet-base-v2 --train_method mulneg --template_id pro/con --train_maps_size 1 --train_per_map_size 8 --data_samples_seed 13 --use_templates True --train_batch_size 8 --output_dir_prefix results/few-shot-seeds-mulneg/all-mpnet-mulneg-pro_con-n8in1-seed13'.split(' ')

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
# sys.argv += '--train_method cossco'.split(' ')

# sys.argv += '--eval_model_name_or_path sentence-transformers/paraphrase-albert-small-v2'.split(' ')
sys.argv += '--do_train False'.split(' ')

# sys.argv += '--rerank True'.split(' ')

sys.argv += '--annotated_samples_in_test True --debug_map_index 17763'.split(' ')

# eval annotated samples
sys.argv += '--do_eval False'.split(' ')
sys.argv += '--do_eval_annotated_samples True --debug_maps_size 3'.split(' ')

# save embeddings
sys.argv += '--save_embeddings True'.split(' ')

# debug specific map
# sys.argv += '--no_data_split True --debug_map_index 22187'.split(' ')

# philosophy domain
# sys.argv += '--training_domain_index 16 --debug_maps_size 1000'.split(' ')

train_triplets_kialo.main()
