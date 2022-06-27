import sys
import train_triplets_kialo


def baseline():
    sys.argv += '--eval_model_name_or_path all-mpnet-base-v2 --eval_not_trained True --do_train False'.split(' ')


sys.argv += ['--local', 'True']
sys.argv += ['--debug_maps_size', '10']
baseline()
train_triplets_kialo.main()
