import os
import urllib.request

from utils.paths import PROJECT_ROOT

MODEL_BEST_SMALL = 'kata1-b6c96-s175395328-d26788732.txt.gz' # 9959.8 ± 16.9
MODEL_9X9 = "kata9x9-b18c384nbt-20231025.bin.gz"
MODEL_HUMAN = "b18c384nbt-humanv0.bin.gz"
# The transformer network: the engine every step starts, and — as its PyTorch
# checkpoint, ACTIVATION_NETWORK — what the activation store is read from. The
# engine needs v1.17 or later to load it.
MODEL_TRANSFORMER = "kata1-tf3-b10c512-s3203M-d5937M.bin.gz"
ACTIVATION_NETWORK = "kata1-tf3-b10c512-s3203M-d5937M"
# The residual network. It drives nothing; it is a second strong-policy bot
# beside the transformer in pipeline.benchmark_bots.
MODEL_RESIDUAL = "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz" # 13617.9 ± 14.9
# The network every engine-driving step starts unless told otherwise. It lives
# here rather than in utils/constants.py because only the KataGo side reads it,
# and constants importing it from here would load this package into every
# rating-side process.
DEFAULT_MODEL = MODEL_TRANSFORMER
# Anchor the model cache to the repo root, not the process CWD — every sibling
# path in the pipeline is PROJECT_ROOT-relative, so a cwd-relative models/ let
# a step launched from another directory silently re-download ~GBs into a stray
# ./models/.
MODEL_DIRECTORY = os.path.join(PROJECT_ROOT, "models")

URL = "https://media.katagotraining.org/uploaded/networks/models/kata1/"
CHECKPOINT_URL = "https://media.katagotraining.org/uploaded/networks/zips/kata1/"

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
        if filename == MODEL_9X9:
            url = "https://github.com/lightvector/KataGo/releases/download/v1.13.2-kata9x9/kata9x9-b18c384nbt-20231025.bin.gz"
        elif filename == MODEL_HUMAN:
            url = "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz"
        else:
            url = URL + filename

        # Download to a temp path, then atomically rename on success.
        # The fetch writes incrementally, so an interrupted ~hundreds-of-MB
        # fetch would otherwise leave a truncated file at file_path that the
        # existence check above trusts forever — every later run then hands
        # KataGo a corrupt model. On failure, remove the partial and re-raise
        # so the caller sees the error instead of a path to a broken file.
        print(f"downloading model {filename}")
        _download(url, file_path)

    return file_path


def _download(url, file_path):
    """Fetch to a temp path and rename on success (see get_model). The request
    names a browser-style agent: the network host refuses urllib's default."""
    import shutil
    tmp_path = file_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as r, open(tmp_path, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(tmp_path, file_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_checkpoint(name):
    """The PyTorch checkpoint of a kata1 network, downloaded lazily like the
    engine files. The loader needs this form — KataGo converts PyTorch to its
    binary format and not back — and the host publishes it as a zip holding
    one `<name>.ckpt`."""
    import zipfile
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    file_path = os.path.join(MODEL_DIRECTORY, f"{name}.ckpt")
    if not os.path.exists(file_path):
        zip_path = os.path.join(MODEL_DIRECTORY, f"{name}.zip")
        print(f"downloading checkpoint {name}")
        _download(CHECKPOINT_URL + f"{name}.zip", zip_path)
        tmp_path = file_path + ".part"
        with zipfile.ZipFile(zip_path) as z, z.open(f"{name}.ckpt") as src, \
                open(tmp_path, "wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst, 1 << 20)
        os.replace(tmp_path, file_path)
        os.remove(zip_path)
    return file_path


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
