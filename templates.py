args: dict[str]
TEMPLATES = {'standard': {'child': 'This sentence: "{}" is child',
                          'parent': 'This sentence: "{}" is parent'},
             'foo': {'child': 'foo: "{}"',
                     'parent': 'bar: "{}"'},
             'beginning': {'child': 'child: "{}"',
                           'parent': 'parent: "{}"'}
             }

def format(text: str, type: str, use_templates: bool):
    if not use_templates:
        return text
    return TEMPLATES[args['template_id']][type].format(text)
