from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.prepvis_metrics import PrepvisMetrics

parser = ArgumentParser(description="QA for MWA UVFITS visibility data")
parser.add_argument('uvfits', type=Path, help='UVFITS visibility file')
parser.add_argument('--out', help='json output path', type=str,
                    default=None, required=False)

args = parser.parse_args()
m = PrepvisMetrics(str(args.uvfits))
m.run_metrics()
m.write_to(args.out)
