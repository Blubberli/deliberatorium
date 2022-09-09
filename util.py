import re


def remove_url_and_hashtags(text):
    pattern = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    match = re.findall(pattern, text)
    for m in match:
        url = m[0]
        text = text.replace(url, '')
    text = text.replace("()", "")
    text = text.replace("#", "")
    return text
