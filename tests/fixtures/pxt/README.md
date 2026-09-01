# Synthetic PXT fixtures

These files contain only deterministic arrays created for peaksMCP tests; they
do not contain experimental measurements or user metadata.

- `synthetic_2d_nested.pxt` contains one 2D ARPES Wave inside nested Igor
  folders. Its energy coordinate is descending.
- `synthetic_3d.PXT` contains a 3D ARPES mapping Wave and uses an upper-case
  extension intentionally.

They were generated with `generate_fixtures.py` and IgorWriter 0.7.1. The
runtime reads them with Igor2, exactly as it reads instrument-produced PXT
files. IgorWriter is needed only when regenerating the fixtures.

The supported instrument contract is exactly one data Wave per PXT. The
loader rejects an unexpected multi-Wave file instead of guessing which Wave
is the experiment data.
