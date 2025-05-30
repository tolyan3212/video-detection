import argparse
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import numpy as np


def draw_box(
    frame: np.ndarray,
    box: Boxes,
    label: str,
) -> None:
    '''Draws a box of a detected object, a label and object's
    confidence on a given frame.

    Args:
        frame (np.ndarray): A frame to be modified.
        box (ultralytics.engine.results.Boxes): An output box of a
            detected object provided by a YOLO model.
        label (str): A label string to put above the box.

    Returns:
        None
    '''
    conf = round(box.conf[0].item(), 3)
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f'{label} {conf}',
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 0),
        2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='video/crowd.mp4')
    parser.add_argument('-o', '--output', default='crowd_detected.mp4')
    args = parser.parse_args()

    # Initialize model
    model = YOLO('yolo12n.pt')

    # Open the input video
    cap = cv2.VideoCapture(args.filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up output video writer
    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model.track(
            frame,
            persist=True,
            tracker='botsort.yaml'
        )[0]

        # Draw objects of the 'person' class
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # class 0 stands for 'person'
                draw_box(frame, box, 'Person')

        out.write(frame)

    cap.release()
    out.release()


if __name__ == '__main__':
    main()
