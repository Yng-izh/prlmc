"""Passive dataset audit; no model, questions, or answers are evaluated."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def templates(path):
    tree = ast.parse(Path(path).read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'TEMPLATES'
                        for t in node.targets))


def inventory(directory, template_path):
    import pandas as pd
    patterns = [re.compile(p) for p, _, _ in templates(template_path)]
    results = []
    for path in sorted(Path(directory).glob('*.parquet')):
        frame = pd.read_parquet(path)
        groups, rows = {}, []
        for index, record in frame.iterrows():
            context = record['context']
            digest = hashlib.sha256(context.encode()).hexdigest()
            facts = [re.sub(r'^[0-9]+\. ', '', line).strip()
                     for line in context.splitlines() if re.match(r'^[0-9]+\. ', line)]
            fact_hash = hashlib.sha256('\n'.join(facts).encode()).hexdigest() if facts else None
            groups.setdefault(digest, []).append(int(index))
            metadata = record.get('metadata')
            scalars = {str(k): str(v) for k, v in metadata.items()
                       if isinstance(v, (str, int, float, bool))} if isinstance(metadata, dict) else {}
            rows.append(dict(row=int(index), context_sha256=digest,
                             fact_sha256=fact_hash, chars=len(context), fact_lines=len(facts),
                             template_matches=sum(any(p.match(f.rstrip('.')) for p in patterns) for f in facts),
                             questions=len(record['questions']), metadata=scalars))
        results.append(dict(file=path.name, columns=list(frame.columns), rows=rows,
                            unique_contexts=len(groups), context_groups=list(groups.values())))
    return dict(status='completed', question_answers_inspected=False, datasets=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--templates', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = inventory(args.data, args.templates)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({d['file']: dict(rows=len(d['rows']), unique=d['unique_contexts'])
                      for d in result['datasets']}))


if __name__ == '__main__':
    main()
