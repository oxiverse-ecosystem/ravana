import sys

def extract(path, start, end):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(start-1, min(end, len(lines))):
        print(f"{i+1}: {lines[i]}", end='')

if __name__ == '__main__':
    path = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    extract(path, start, end)
