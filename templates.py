def format(text: str, type: str, use_templates: bool):
    if not use_templates:
        return text
    return {'child': 'This sentence: "{}" is child',
            'parent': 'This sentence: "{}" is parent'}[type].format(text)
