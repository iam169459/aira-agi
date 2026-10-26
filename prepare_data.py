import os
from pathlib import Path

def collect_local_python_files(root_dir):
    texts = []
    root_path = Path(root_dir)
    for path in root_path.rglob('*.py'):
        if any(part.startswith('.') for part in path.parts):
            continue
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f_in:
                content = f_in.read()
            if len(content.strip()) == 0:
                continue
            texts.append(f"# FILE: {path.relative_to(root_path)}\n{content}\n\n")
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return ''.join(texts)

if __name__ == '__main__':
    corpus = collect_local_python_files('.')
    output_file = 'code_corpus.txt'
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(corpus)
    print(f"Success! Wrote {len(corpus)} characters from local Python files to {output_file}")
