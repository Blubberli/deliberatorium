import re
import random


def remove_url_and_hashtags(text):
    text = re.sub(r'\[([^\]]+)\]\(http[^\)]+\)', r'\1', text)
    text = text.replace("#", "")
    return text


def sample(x: list, max_size: int):
    return random.sample(x, min(max_size, len(x)))
