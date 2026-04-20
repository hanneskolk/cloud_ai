from ultralytics.trackers.byte_tracker import BYTETracker

tracker = BYTETracker()

def track(detections, frame):
    return tracker.update(detections, frame)