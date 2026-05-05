# Demo Video Sources

This directory holds the input footage analyzed by the dashboard.

## Getting a demo clip

Two scripts produce `samples/demo_traffic.mp4`:

### Option A — download a real clip (recommended)

```bash
python scripts/fetch_demo_video.py
```

Downloads a short, CC-licensed public traffic clip from a stable mirror.
Source URL and license are written into `samples/SOURCE.txt` after download.

### Option B — synthesize offline

```bash
python scripts/make_demo_video.py
```

Generates a 15-second 640x360 clip of two colored rectangles representing
"vehicles" colliding mid-frame. Useful when offline or when the download
script cannot reach the network. The synthetic clip exercises the
frame-difference fallback path in `core/anomaly_detector.py` because YOLOv8
will not detect the rectangles as COCO vehicle classes.

### Option C — bring your own

Drop any `.mp4` into this folder, rename it `demo_traffic.mp4`, and the
dashboard will pick it up. Keep clips under ~2 minutes for the MVP — see
`config.MAX_VIDEO_LENGTH_SEC`.

## Suggested public datasets (for stretch goals)

- **BDD100K** — https://bdd-data.berkeley.edu/ (registration required)
- **KITTI** — https://www.cvlibs.net/datasets/kitti/
- **Pixabay traffic videos** — https://pixabay.com/videos/search/traffic/ (CC0)
- **Wikimedia Commons traffic** — https://commons.wikimedia.org/

## Generated artifacts

`demo_run.log` is a captured stdout/stderr of the end-to-end pipeline run.
It is regenerated each time `app.py` is launched in capture mode.
