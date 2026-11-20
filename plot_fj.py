"""Compare path-task evaluations from selected methods and training seeds."""
import argparse,json
from pathlib import Path
from fjbench.config import select_methods


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations',nargs='+',required=True)
    p.add_argument('--methods',help='Optional comma-separated plotted subset')
    p.add_argument('--out',required=True)
    a=p.parse_args();chosen=select_methods(a.methods) if a.methods else None
    reports=[json.loads(Path(s).read_text()) for s in a.evaluations]
    reports=[r for r in reports if r.get('path_metrics') and (chosen is None or r['method'] in chosen)]
    identities=[(r['method'],r['config']['seed']) for r in reports]
    if len(identities)!=len(set(identities)):p.error('Duplicate method/training-seed result')
    from fjbench.reporting import plot_comparison
    plot_comparison(reports,a.out)
    print(a.out)


if __name__=='__main__':main()
