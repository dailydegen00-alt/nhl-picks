content = open('build_html.py', encoding='utf-8').read()

old = "    # O/U row\n    ou_row = {}\n    for feat in features_ou:"
new = "    # O/U row\n    ou_row = {'gsax_diff': h_gsax - a_gsax}\n    for feat in features_ou:"

if old in content:
    content = content.replace(old, new)
    open('build_html.py', 'w', encoding='utf-8').write(content)
    print("Patched successfully!")
else:
    print("Pattern not found - searching for similar...")
    idx = content.find('# O/U row')
    print(repr(content[idx:idx+80]))
