import cv2

def draw_boxes(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2, track_id = map(int, t[:5])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame