import ast, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
try:
    ast.parse(content)
    print(f'OK: {path} parses')
except SyntaxError as e:
    print(f'SYNTAX ERROR at line {e.lineno}: {e.msg}')
    lines = content.split('\n')
    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+3)):
        print(f'{i+1}: {lines[i]}')
    sys.exit(1)
