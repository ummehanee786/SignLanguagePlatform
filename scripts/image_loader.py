"""
image_loader.py

Task 2 - Image Loader

Given any image path, this script:
    - Reads the image
    - Displays it in a window
    - Prints Height, Width, Channels, and Image Size

No preprocessing. No model. Just reading images correctly.
This exact logic becomes part of the preprocessing pipeline later.

Usage:
    python image_loader.py path/to/image.jpg
"""

import argparse
import os
from pathlib import Path

import cv2


def load_and_display(image_path: str):
    path = Path(image_path)

    if not path.exists():
        print(f"[!] File does not exist: {path}")
        return

    if not path.is_file():
        print(f"[!] Path is not a file: {path}")
        return

    # Read the image. cv2.imread returns None (not an exception) if the
    # file can't be read as an image, so that has to be checked explicitly.
    image = cv2.imread(str(path))

    if image is None:
        print(f"[!] Could not read image (unsupported format or corrupt file): {path}")
        return

    # image.shape for a color image is (height, width, channels).
    # For a grayscale image it would only be (height, width) - handle both.
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    file_size_bytes = os.path.getsize(path)

    print(f"\nImage path : {path}")
    print(f"Height     : {height}")
    print(f"Width      : {width}")
    print(f"Channels   : {channels}")
    print(f"File size  : {file_size_bytes} bytes ({file_size_bytes / 1024:.2f} KB)")

    window_name = f"Image Loader - {path.name}"
    cv2.imshow(window_name, image)
    print("\nPress any key on the image window to close it...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Read and display an image, printing its basic properties."
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the image file to load (relative or absolute).",
    )
    args = parser.parse_args()

    image_path = args.image_path

    # If no path was passed on the command line, just ask for one.
    # Makes it much easier to test quickly without typing long commands.
    if not image_path:
        raw = input("Enter the path to an image file: ").strip()

        # PowerShell prefixes drag-and-dropped paths with "& " (the call
        # operator) and wraps them in quotes, e.g.: & 'C:\path\file.jpg'
        # Strip that off so it isn't treated as part of the path.
        if raw.startswith("&"):
            raw = raw[1:].strip()

        image_path = raw.strip('"').strip("'")

    load_and_display(image_path)


if __name__ == "__main__":
    main()