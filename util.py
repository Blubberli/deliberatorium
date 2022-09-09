import re


def remove_url_and_hashtags(text):
    text = re.sub(r'\[([^\]]+)\]\(http[^\)]+\)', r'\1', text)
    text = text.replace("#", "")
    return text
