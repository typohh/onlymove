import os
import urllib.request

MODEL_BEST_SMALL = 'kata1-b6c96-s175395328-d26788732.txt.gz' # 9959.8 ± 16.9
MODEL_FULL = "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz" # 13617.9 ± 14.9
MODEL_9X9 = "kata9x9-b18c384nbt-20231025.bin.gz"
MODEL_HUMAN = "b18c384nbt-humanv0.bin.gz"
MODEL_DIRECTORY = "models"

URL = "https://media.katagotraining.org/uploaded/networks/models/kata1/"

MODELS = [
    "kata1-b6c96-s175395328-d26788732.txt.gz",  # 9959.8 ± 16.9
    "kata1-b10c128-s1141046784-d204142634.txt.gz", # 11515.5 ± 16.1
    "kata1-b15c192-s1672170752-d466197061.txt.gz", # 12213.8 ± 16.1
    "kata1-b20c256x2-s5303129600-d1228401921.bin.gz", # 12903.7 ± 18.7
    "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",  # 13617.9 ± 14.9
]


def get_model(filename):
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

    file_path = os.path.join(MODEL_DIRECTORY, filename)

    if not os.path.exists(file_path):

        def download_file(url, filename):
            try:
                print(f"downloading model {filename}")
                urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")


        if filename == MODEL_9X9:
            url = "https://github.com/lightvector/KataGo/releases/download/v1.13.2-kata9x9/kata9x9-b18c384nbt-20231025.bin.gz"
        elif filename == MODEL_HUMAN:
            url = "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz"
        else:
            url = URL + filename

        download_file(url, file_path)

    return os.path.join(MODEL_DIRECTORY, filename)


def get_filename_without_extension(path):
    """Extract filename without any extensions from a given path.

    Args:
        path: String containing file path and name with possible multiple extensions

    Returns:
        String containing only the filename without any extensions
    """
    basename = os.path.basename(path)
    filename = basename.split('.')[0]
    return filename
