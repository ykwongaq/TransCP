import os
import numpy as np

import argparse
import torch
import json

from typing import List, Dict


def read_json(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def convert(json_file: str, output_file: str) -> None:
    data = read_json(json_file)

    IMAGE_ID_TO_ANNOTATION = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in IMAGE_ID_TO_ANNOTATION:
            IMAGE_ID_TO_ANNOTATION[image_id] = []
        IMAGE_ID_TO_ANNOTATION[image_id].append(annotation)

    data_list = []
    for image_data in data["images"]:
        image_id = image_data["id"]
        image_filename = os.path.basename(image_data["file_name"])
        annotations = IMAGE_ID_TO_ANNOTATION.get(image_id, [])
        for annotation in annotations:
            bbox = annotation["bbox"]
            x, y, w, h = bbox
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
            phase = annotation["caption"]
            data = [image_filename, [], [x_min, y_min, x_max, y_max], phase, []]
            data_list.append(data)

    torch.save(data_list, output_file)


def main(args):
    annotation_folder = args.ann_folder
    output_folder = args.output_folder

    print(f"Annotation folder: {annotation_folder}")
    print(f"Output folder: {output_folder}")

    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for annotation_filename in os.listdir(annotation_folder):
        if not annotation_filename.endswith(".json"):
            continue

        print(f"Converting {annotation_filename} ...")
        annotation_file = os.path.join(annotation_folder, annotation_filename)
        output_filename = annotation_filename.replace(".json", ".pth")
        output_file = os.path.join(output_folder, f"marinedet_{output_filename}")
        convert(annotation_file, output_file)
        print(f"Converted {annotation_filename} to {output_file}")


if __name__ == "__main__":
    DEFAULT_ANN_FOLDER = "/mnt/hdd/davidwong/data/marinedet/annotations"
    DEFAULT_OUTPUT_FOLDER = "/mnt/hdd/davidwong/data/VLTVG/split/data/marinedet"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_folder",
        type=str,
        default=DEFAULT_ANN_FOLDER,
        help=f"Path to the annotation folder. Default: {DEFAULT_ANN_FOLDER}",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Path to the output folder. Default: {DEFAULT_OUTPUT_FOLDER}",
    )
    args = parser.parse_args()
    main(args)
