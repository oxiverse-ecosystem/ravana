import ast
try:
    with open('ravana/src/ravana/chat/engine_self_query.py') as f:
        ast.parse(f.read())
    print('SYNTAX OK')
except SyntaxError as e:
    print(f'SYNTAX ERROR: {e}')
