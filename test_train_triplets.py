import logging

import train_triplets


def mock_parse_args(overwrite_args=None):
    if overwrite_args is None:
        overwrite_args = {}
    default_args = {'model_name_or_path': 'xlm-roberta-base',
                    'eval_model_name_or_path': None,
                    'local': True,
                    'do_train': False,
                    'do_eval': False,
                    'lang': '*',
                    'argument_map': '*'}
    train_triplets.parse_args = lambda: {**default_args,
                                         **overwrite_args}


def test_all_maps(caplog):
    mock_parse_args()
    caplog.set_level(logging.INFO)
    train_triplets.main()
    log_text = caplog.text
    print(log_text)
    for x in train_triplets.AVAILABLE_MAPS:
        assert x in log_text


def test_one_maps(caplog):
    mock_parse_args({'argument_map': 'doppariam1'})
    caplog.set_level(logging.INFO)
    train_triplets.main()
    log_text = caplog.text
    print(log_text)
    assert 'doppariam1' in log_text
    assert 'dopariam2' not in log_text
