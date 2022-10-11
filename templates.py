import util

TEMPLATES = {
    'beginning': {'child': 'child: "{}"',
                  'parent': 'parent: "{}"'},
    'foo': {'child': 'foo: "{}"',
            'parent': 'bar: "{}"'},
    'detailed': {'child': 'This sentence: "{}" is child',
                 'parent': 'This sentence: "{}" is parent'},
    'pro/con': {'pro': 'pro: "{}"',
                'con': 'contra: "{}"'},
    'combined': {'combined': '"{}" is a child of "{}"'},
}

standard_template = 'beginning'
for k, v in TEMPLATES.items():
    if 'child' not in v:
        v['child'] = TEMPLATES[standard_template]['child']
    if 'parent' not in v:
        v['parent'] = TEMPLATES[standard_template]['parent']
    v['possible_templates'] = {'parent': ['parent'], 'child': ['child'], 'pro': ['child'], 'con': ['child']}
    if 'pro' in v:
        v['possible_templates']['pro'].append('pro')
        v['possible_templates']['con'].append('con')
    if 'combined' in v:
        for kk, vv in v['possible_templates'].items():
            if kk != 'parent':
                v['possible_templates'][kk].append('combined')


# primary template is used as main template for eval
def format_primary(text: str, node_type: str, use_templates: bool):
    if not use_templates:
        return text
    return TEMPLATES[util.args['template_id']][node_type].format(text)


def format_all_possible(text: str, parent_text: str, node_type: str, use_templates: bool):
    if not use_templates:
        return [text]
    return [TEMPLATES[util.args['template_id']][t].format(*([text, parent_text] if t == 'combined' else [text])) for t in
            TEMPLATES[util.args['template_id']]['possible_templates'][node_type]]
