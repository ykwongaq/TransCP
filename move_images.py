import argparse
import os
from tqdm import tqdm


def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir

    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Move the image to the output directory
        if os.path.islink(output_path):
            os.unlink(output_path)

        os.symlink(image_path, output_path)


if __name__ == "__main__":
    DEFAULT_IMAGE_DIR = "/mnt/hdd/davidwong/data/marinedet/images"
    DEFAULT_OUTPUT_DIR = "/mnt/hdd/davidwong/data/VLTVG/data/marinedet"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help=f"Path to the image directory. Default: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to the output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    args = parser.parse_args()
    main(args)
