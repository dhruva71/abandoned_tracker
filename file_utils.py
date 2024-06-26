from typing import Any

import cv2


def save_frame(output_dir: str, file_name: str, frame: Any):
    frame_path = f'{output_dir}/{file_name}'
    cv2.imwrite(str(frame_path), frame)

    print(f"Frame saved at {frame_path}")
    return frame_path
