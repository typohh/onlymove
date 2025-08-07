import os
import urllib.request

MODEL_BEST_SMALL = 'kata1-b6c96-s175395328-d26788732.txt.gz'
MODEL_FULL = "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
MODEL_9X9 = "kata9x9-b18c384nbt-20231025.bin.gz"
MODEL_HUMAN = "b18c384nbt-humanv0.bin.gz"
MODEL_DIRECTORY = "models"


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

        urls = {
            MODEL_9X9: "https://github.com/lightvector/KataGo/releases/download/v1.13.2-kata9x9/kata9x9-b18c384nbt-20231025.bin.gz",
            MODEL_HUMAN: "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz",
            MODEL_FULL: "https://media.katagotraining.org/uploaded/networks/models/kata1/" + MODEL_FULL,
            MODEL_BEST_SMALL: "https://media.katagotraining.org/uploaded/networks/models/kata1/" + MODEL_BEST_SMALL,
        }

        download_file(urls[filename], file_path)

    return os.path.join(MODEL_DIRECTORY, filename)


def get_corrupted_model(scale):
    """Creates a corrupted version of WEIGHT_BEST_SMALL model by multiplying values
    with random samples from log normal distribution.

    Args:
        scale: Float value controlling the amount of corruption
    """
    import numpy as np
    import gzip

    # seed it based on scale, so minor changes in scale alter the "personality"
    np.random.seed(round(1000 * scale))

    input_path = os.path.join(MODEL_DIRECTORY, MODEL_BEST_SMALL)
    output_filename = get_filename_without_extension(input_path) + f'_corrupt{int(scale * 100)}.txt.gz'
    output_path = os.path.join(MODEL_DIRECTORY, output_filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Model file not found: {input_path}")

    if os.path.exists(output_path):
        return output_path

    print("creating model ", output_filename)

    # Parameters for log normal distribution
    sigma = np.sqrt(np.log(1 + scale ** 2))
    mu = -np.log(1 + scale ** 2) / 2

    with gzip.open(input_path, 'rt') as f_in, gzip.open(output_path, 'wt') as f_out:
        for line in f_in:

            # Split line into parts
            parts = line.strip().split()

            def is_float(string):
                try:
                    float(string)
                    return '.' in string
                except ValueError:
                    return False

            # Process only lines containing float numbers
            if len(parts) > 1 and is_float(parts[0]):
                # Multiply each number by random sample
                corrupted = [str(float(x) * np.random.lognormal(mu, sigma))
                             for x in parts]
                f_out.write(' '.join(corrupted) + '\n')
            else:
                f_out.write(line)

    return output_path


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
