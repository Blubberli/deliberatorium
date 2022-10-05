import random
import re
from typing import Union, Sequence, AbstractSet


def remove_url_and_hashtags(text):
    text = re.sub(r'\[([^\]]+)\]\(http[^\)]+\)', r'\1', text)
    text = text.replace("#", "")
    return text


def sample(x: Union[Sequence, AbstractSet], max_size: int):
    return random.sample(x, min(max_size, len(x)))
