from ultralytics.trackers.byte_tracker import BYTETracker
from types import SimpleNamespace

args = SimpleNamespace(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=30
)

tracker = BYTETracker(args)

def track(detections, frame):
    return tracker.update(detections, frame)