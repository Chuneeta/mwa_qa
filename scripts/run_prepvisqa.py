from argparse import ArgumentParser
from pathlib import Path
from mwa_qa.prepvis_metrics import PrepvisMetrics

parser = ArgumentParser(description="QA for MWA UVFITS visibility data")
parser.add_argument('uvfits', type=Path, help='UVFITS visibility file')
parser.add_argument('metafits', type=Path, help='Metafits file')
parser.add_argument('--ex_annumbers', type=list, default=[],
                    help='List of antennas to be excluded. Default is empty')
parser.add_argument('--edge_flagging', action='store_true',
                    help='Flags frequency edge and centre channle of each coarse bands. Default is True')
parser.add_argument('--antenna_flags', action='store_true',
                    help='Apply flags to antenna array. Default is False')
parser.add_argument('--cutoff_threshold', type=float, default=3,
                    help='Cutoff threshold value for modified z-score to identify outliers.')
parser.add_argument('--niter', type=int, default=10,
                    help='Number of iteration for outlier identification using modified z-score.')
parser.add_argument('--out', help='json output path', type=str,
                    default=None, required=False)
parser.add_argument('--split', action='store_true', help='Allow for splitting of autocorrelations based on antennas naming convention',
                    default=None, required=False)

args = parser.parse_args()
m = PrepvisMetrics(str(args.uvfits), str(args.metafits),
                   ex_annumbers=args.ex_annumbers, edge_flagging=args.edge_flagging,
                   antenna_flags=args.antenna_flags, cutoff_threshold=args.cutoff_threshold,
                   niter=args.niter)
m.run_metrics(args.split)
m.write_to(args.out)
