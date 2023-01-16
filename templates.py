import copy
import itertools

import util

UNIQUE_TEMPLATES = {
    'beginning': {'child': 'child: "{}"',
                  'parent': 'parent: "{}"'},
    'foo': {'child': 'foo: "{}"',
            'parent': 'bar: "{}"'},
    'pro/con': {'pro': 'pro: "{}"',
                'con': 'contra: "{}"'},
    'combined': {'child': 'child: "{}" parent: "{}"'},
    'combined_pro/con': {'pro': 'pro: "{}" parent: "{}"',
                         'con': 'contra: "{}" parent: "{}"'},
    'all': {},
    'all-meaningless': {}
}

MEANINGLESS_UNIQUE_TEMPLATES = {
    'foo': {'child': 'foo: "{}"',
            'parent': 'bar: "{}"'},
    'beginning': {'child': 'baz: "{}"',
                  'parent': 'qux: "{}"'},
    'pro/con': {'pro': 'grault: "{}"',
                'con': 'garply: "{}"'},
    'combined': {'child': '"{}" waldo "{}"'},
    'combined_pro/con': {'pro': 'grault: "{}" qux: "{}"',
                         'con': 'garply: "{}" qux: "{}"'},
    'all': {},
    'all-meaningless': {}
}

POSSIBLE_TEMPLATES = {
    'parent': ['parent'],
    'pro': ['child', 'pro'],
    'con': ['child', 'con']
}

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
    template_id = util.args['template_id'] if util.args['template_id'] != 'all-meaningless' else 'foo'
    return TEMPLATES[template_id][node_type].format(text)


def format_all_possible(text: str, parent_text: str, node_type: str, use_templates: bool, parent=True):
    if not use_templates:
        return [text]

    if not parent:
        text = f'not {text}'

    if util.args['template_id'] in ('all', 'all-meaningless'):
        if node_type == 'parent':
            # use one parent representation to avoid repetition of same parent in the same batch
            return [format_primary(text, node_type, use_templates)]
        else:
            return list(itertools.chain.from_iterable(
                format_using_template(
                    text, parent_text, node_type, t,
                    UNIQUE_TEMPLATES if util.args['template_id'] == 'all' else MEANINGLESS_UNIQUE_TEMPLATES)
                for t in UNIQUE_TEMPLATES.keys()))
    else:
        return format_using_template(text, parent_text, node_type, util.args['template_id'], TEMPLATES)


def format_using_template(text: str, parent_text: str, node_type: str, template_id: str, templates: dict):
    return [templates[template_id][t].format(*([text, parent_text] if 'combined' in template_id else [text]))
            for t in POSSIBLE_TEMPLATES[node_type] if t in templates[template_id]]


util.args = {}
util.args['template_id'] = 'all'
# util.args['template_id'] = 'all-meaningless'

print('child')
print(*[x for x in format_all_possible('child_text', 'parent_text', 'pro', True, True)], sep='\n')
print('parent')
print(*[x for x in format_all_possible('parent_text', 'grand_text', 'parent', True, True)], sep='\n')
