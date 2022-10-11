import copy
import itertools

import util

UNIQUE_TEMPLATES = {
    'beginning': {'child': 'child: "{}"',
                  'parent': 'parent: "{}"'},
    'foo': {'child': 'foo: "{}"',
            'parent': 'bar: "{}"'},
    'detailed': {'child': 'This sentence: "{}" is child',
                 'parent': 'This sentence: "{}" is parent'},
    'pro/con': {'pro': 'pro: "{}"',
                'con': 'contra: "{}"'},
    'combined': {'combined': '"{}" is a child of "{}"'},
    'all': {}
}

POSSIBLE_TEMPLATES = {'parent': ['parent'], 'pro': ['child', 'pro', 'combined'], 'con': ['child', 'con', 'combined']}

standard_template = 'beginning'
TEMPLATES = copy.deepcopy(UNIQUE_TEMPLATES)
for k, v in TEMPLATES.items():
    if 'child' not in v:
        v['child'] = TEMPLATES[standard_template]['child']
    if 'parent' not in v:
        v['parent'] = TEMPLATES[standard_template]['parent']


# primary template is used as main template for eval
def format_primary(text: str, node_type: str, use_templates: bool):
    if not use_templates:
        return text
    return TEMPLATES[util.args['template_id']][node_type].format(text)


def format_all_possible(text: str, parent_text: str, node_type: str, use_templates: bool):
    if not use_templates:
        return [text]

    if util.args['template_id'] == 'all':
        if node_type == 'parent':
            # use one parent representation to avoid repetition of same parent in the same batch
            return [format_primary(text, node_type, use_templates)]
        else:
            return list(itertools.chain.from_iterable(
                format_using_template(text, parent_text, node_type, t, UNIQUE_TEMPLATES) for t in UNIQUE_TEMPLATES.keys()))
    else:
        return format_using_template(text, parent_text, node_type, util.args['template_id'], TEMPLATES)


def format_using_template(text: str, parent_text: str, node_type: str, template_id: str, templates: dict):
    return [templates[template_id][t].format(*([text, parent_text] if t == 'combined' else [text]))
            for t in POSSIBLE_TEMPLATES[node_type] if t in templates[template_id]]
