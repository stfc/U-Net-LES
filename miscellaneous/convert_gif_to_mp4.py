import os
import sys
from moviepy.editor import VideoFileClip


def convert_gif_to_mp4(gif_path):
    """Converts a large GIF file to a compressed MP4 video."""
    # Verify the file exists before processing
    if not os.path.exists(gif_path):
        print(f"Error: The file '{gif_path}' does not exist.")
        sys.exit(1)

    # Check if the input file actually has a .gif extension
    if not gif_path.lower().endswith(".gif"):
        print("Warning: The input file does not have a .gif extension.")

    # Generate the output MP4 filename in the same folder
    mp4_path = os.path.splitext(gif_path)[0] + ".mp4"

    print(f"Loading '{gif_path}'...")

    try:
        # Load and convert the clip
        clip = VideoFileClip(gif_path)
        print(f"Converting and compressing to '{mp4_path}'...")

        clip.write_videofile(
            mp4_path,
            codec="libx264",
            audio=False,  # GIFs do not have audio
            logger="bar",  # Displays a progress bar
        )

        clip.close()
        print("Conversion completed successfully!")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if the user passed a filename argument
    if len(sys.argv) < 2:
        print("Usage: python gif_to_mp4.py <path_to_gif_file>")
        sys.exit(1)

    # Get the file path from the first command-line argument
    input_file = sys.argv[1]
    convert_gif_to_mp4(input_file)

