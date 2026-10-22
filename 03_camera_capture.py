"""
Step 3 -- Capture RGB, depth and (optional) segmentation images from every drone.

Produces the observation data stream a MARL policy will later consume:
  ./captures/<drone>/rgb_<n>.png          scene camera
  ./captures/<drone>/depth_<n>.png        depth visualised in grayscale
  ./captures/<drone>/seg_<n>.png          ground-truth instance segmentation

Requires:  pip install opencv-python numpy
"""
import os

import airsim
import cv2
import numpy as np

DRONES = ["Drone1", "Drone2", "Drone3", "Drone4"]
OUT_DIR = "captures"


def save_response(resp, path):
    if resp.pixels_as_float:
        arr = airsim.get_pfm_array(resp)          # float depth
        arr = np.clip(arr, 0, 50)                 # clip at 50 m for readability
        img = (255 * arr / 50.0).astype(np.uint8)
        cv2.imwrite(path, img)
    else:
        cv2.imwrite(path, airsim.string_to_uint8_array(resp.image_data_uint8))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = airsim.MultirotorClient()
    client.confirmConnection()

    for d in DRONES:
        drone_dir = os.path.join(OUT_DIR, d)
        os.makedirs(drone_dir, exist_ok=True)

        responses = client.simGetImages(
            [
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
            ],
            vehicle_name=d,
        )
        tags = ["rgb", "depth", "seg"]
        for tag, resp in zip(tags, responses):
            path = os.path.join(drone_dir, f"{tag}_0.png")
            save_response(resp, path)
            print(f"[OK] {d}: saved {path}  ({resp.width}x{resp.height})")

    print("\nAll captures written to", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
