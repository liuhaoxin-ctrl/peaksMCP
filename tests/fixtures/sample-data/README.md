# Sample-data mirror

Local copies of the Peaks `ExampleData` files that the test suites need.
Keeping them here avoids re-downloading from Zenodo on every fresh test
process (the downloader has no persistent cache).

Set the mirror so `peaks.core.utils.sample_data.ZenodoDownloader` finds the
files locally instead of downloading:

```bash
export LOCAL_MIRROR_PATH=/abs/path/to/peaksMCP/tests/fixtures/sample-data
```

- `i05-59819.nxs`            — `ExampleData.dispersion()`
- `210326_GM2-667_GK_1.xy`   — `ExampleData.dispersion2a()`
- `Ep20eV.xy`                — `ExampleData.gold_reference4()`

Regenerate/add files with the same names via `ZenodoDownloader` if the source
upstream data changes.