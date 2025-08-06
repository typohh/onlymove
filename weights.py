WEIGHTS = {
    'kata1-b15c192-s1672170752-d466197061.txt.gz': 12212,
    'kata1-b10c128-s1141046784-d204142634.txt.gz': 11514,
    'kata1-b6c96-s175395328-d26788732.txt.gz': 9958,
    # 'kata1-b6c96-s165180416-d25130434.txt.gz': 9900,
    # 'kata1-b6c96-s152505856-d23152636.txt.gz': 9833,
    # 'kata1-b6c96-s141089536-d21243181.txt.gz': 9793,
    # 'kata1-b6c96-s130521856-d19523626.txt.gz': 9740,
    'kata1-b6c96-s122554112-d18283013.txt.gz': 9686,
    # 'kata1-b6c96-s115648256-d17228996.txt.gz': 9634,
    # 'kata1-b6c96-s108812288-d16129874.txt.gz': 9567,
    # 'kata1-b6c96-s103950080-d15368530.txt.gz': 9515,
    'kata1-b6c96-s95425792-d14091095.txt.gz': 9416,
    'kata1-b6c96-s89508352-d13124432.txt.gz': 9350,
    'kata1-b6c96-s83588096-d12203675.txt.gz': 9251,
    'kata1-b6c96-s78659584-d11478487.txt.gz': 9129,
    'kata1-b6c96-s73091584-d10630987.txt.gz': 9002,
    'kata1-b6c96-s69427456-d10051148.txt.gz': 8925,
    'kata1-b6c96-s65341184-d9428755.txt.gz': 8808,
    'kata1-b6c96-s63439616-d9138325.txt.gz': 8721,
    'kata1-b6c96-s59560704-d8589539.txt.gz': 8589,
    'kata1-b6c96-s56669696-d8173106.txt.gz': 8452,
    'kata1-b6c96-s55682304-d7995124.txt.gz': 8416,
    'kata1-b6c96-s52864512-d7635581.txt.gz': 8280,
    'kata1-b6c96-s50894592-d7380655.txt.gz': 8192,
    'kata1-b6c96-s48921344-d7092247.txt.gz': 8084,
    'kata1-b6c96-s46949632-d6822967.txt.gz': 7962,
    'kata1-b6c96-s45189632-d6589032.txt.gz': 7822,
    'kata1-b6c96-s43286272-d6326857.txt.gz': 7666,
    'kata1-b6c96-s41312768-d6061202.txt.gz': 7516,
    'kata1-b6c96-s39339776-d5828519.txt.gz': 7303,
    'kata1-b6c96-s37368064-d5536083.txt.gz': 7096,
    'kata1-b6c96-s35356160-d5299514.txt.gz': 6838,
    'kata1-b6c96-s33239552-d5016295.txt.gz': 6555,
    'kata1-b6c96-s32344832-d4917369.txt.gz': 6475,
    'kata1-b6c96-s31390976-d4781382.txt.gz': 6322,
    'kata1-b6c96-s30394624-d4635675.txt.gz': 6172,
    'kata1-b6c96-s29422336-d4533650.txt.gz': 5987,
    'kata1-b6c96-s28460288-d4388379.txt.gz': 5768,
    'kata1-b6c96-s27415296-d4256697.txt.gz': 5614,
    'kata1-b6c96-s26363136-d4119819.txt.gz': 5392,
    'kata1-b6c96-s25461504-d4002500.txt.gz': 5145,
    'kata1-b6c96-s24455424-d3879081.txt.gz': 4925,
    'kata1-b6c96-s23401472-d3742377.txt.gz': 4765,
    'kata1-b6c96-s22371328-d3612586.txt.gz': 4597,
    'kata1-b6c96-s21371392-d3529696.txt.gz': 4439,
    'kata1-b6c96-s20302336-d3388395.txt.gz': 4261,
    'kata1-b6c96-s19408128-d3280178.txt.gz': 4161,
    'kata1-b6c96-s18429184-d3197121.txt.gz': 3942,
    'kata1-b6c96-s17434112-d3038653.txt.gz': 3697,
    'kata1-b6c96-s16525312-d2925067.txt.gz': 3476,
    'kata1-b6c96-s15704832-d2832803.txt.gz': 3321,
    'kata1-b6c96-s14649344-d2727367.txt.gz': 3094,
    'kata1-b6c96-s13733120-d2631546.txt.gz': 2848,
    'kata1-b6c96-s12849664-d2510774.txt.gz': 2544,
    'kata1-b6c96-s11888896-d2416753.txt.gz': 2398,
    'kata1-b6c96-s10825472-d2300510.txt.gz': 2293,
    'kata1-b6c96-s10014464-d2201128.txt.gz': 2112,
    'kata1-b6c96-s8982784-d2082583.txt.gz': 1941,
    'kata1-b6c96-s8080640-d1961030.txt.gz': 1862,
    'kata1-b6c96-s6127360-d1754797.txt.gz': 1710,
    'kata1-b6c96-s5214720-d1690538.txt.gz': 1611,
    'kata1-b6c96-s4136960-d1510003.txt.gz': 1539,
    'kata1-b6c96-s1995008-d1329786.txt.gz': 1071,
    'kata1-b6c96-s938496-d1208807.txt.gz': 800,
    'kata1-b6c96-s1248000-d550347.txt.gz': 482
}

WEIGHT_BEST_SMALL = 'kata1-b6c96-s175395328-d26788732.txt.gz'
WEIGHT_FULL = "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
WEIGHT_9X9 = "kata9x9-b18c384nbt-20231025.bin.gz"
WEIGHT_HUMAN = "b18c384nbt-humanv0.bin.gz"

WEIGHT_DIRECTORY = "weights"


def download_weights():
    import os
    import urllib.request

    os.makedirs(WEIGHT_DIRECTORY, exist_ok=True)

    def download_file(url, filename):
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, os.path.join(WEIGHT_DIRECTORY, filename))
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    # Download base weights if missing
    base_weights = {
        WEIGHT_9X9: "https://github.com/lightvector/KataGo/releases/download/v1.13.2-kata9x9/kata9x9-b18c384nbt-20231025.bin.gz",
        WEIGHT_HUMAN: "https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz",
        WEIGHT_FULL: "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
    }

    for filename, url in base_weights.items():
        if not os.path.exists(os.path.join(WEIGHT_DIRECTORY, filename)):
            download_file(url, filename)

    # Download training weights if missing
    base_url = "https://media.katagotraining.org/uploaded/networks/models/kata1/"
    for weight_file in WEIGHTS.keys():
        if not os.path.exists(os.path.join(WEIGHT_DIRECTORY, weight_file)):
            download_file(base_url + weight_file, weight_file)


def create_corrupted(scale, filename):
    """Creates a corrupted version of WEIGHT_BEST_SMALL model by multiplying values
    with random samples from log normal distribution.

    Args:
        scale: Float value controlling the amount of corruption
    """
    import os
    import numpy as np
    import gzip

    # seed it based on scale, so minor changes in scale alter the "personality"
    np.random.seed(round(1000 * scale))

    input_path = os.path.join(WEIGHT_DIRECTORY, WEIGHT_BEST_SMALL)
    output_path = os.path.join(WEIGHT_DIRECTORY, filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Weight file not found: {input_path}")

    if os.path.exists(output_path):
        return

    print("creating ", filename)

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


WEIGHTS_CORRUPTED = {
    'corrupt2.txt.gz': 2,
    'corrupt5.txt.gz': 5,
    'corrupt7.txt.gz': 7,
    'corrupt10.txt.gz': 10,
    'corrupt12.txt.gz': 12,
    'corrupt15.txt.gz': 15,
    'corrupt17.txt.gz': 17,
    'corrupt20.txt.gz': 20,
    'corrupt22.txt.gz': 22,
    'corrupt25.txt.gz': 25,
    'corrupt27.txt.gz': 27,
    'corrupt28.txt.gz': 28,
    'corrupt29.txt.gz': 29,
    'corrupt30.txt.gz': 30,
    'corrupt305.txt.gz': 30.5,
    'corrupt31.txt.gz': 31,
    'corrupt315.txt.gz': 31.5,
    'corrupt32.txt.gz': 32,
    'corrupt325.txt.gz': 32.5,
    'corrupt33.txt.gz': 33,
    'corrupt335.txt.gz': 33.5,
    'corrupt34.txt.gz': 34,
    'corrupt345.txt.gz': 34.5,
    'corrupt35.txt.gz': 35,
    'corrupt355.txt.gz': 35.5,
    'corrupt36.txt.gz': 36,
    'corrupt365.txt.gz': 36.5,
    'corrupt37.txt.gz': 37,
    'corrupt375.txt.gz': 37.5,
    'corrupt38.txt.gz': 38,
    'corrupt39.txt.gz': 39,
    'corrupt40.txt.gz': 40,
    'corrupt42.txt.gz': 42,
    'corrupt45.txt.gz': 45,
    'corrupt47.txt.gz': 47,
    'corrupt50.txt.gz': 50,
    'corrupt52.txt.gz': 52,
    'corrupt55.txt.gz': 55,
    'corrupt57.txt.gz': 57,
    'corrupt60.txt.gz': 60,
    'corrupt65.txt.gz': 65,
}

if __name__ == "__main__":
    download_weights()
    for filename, percent in WEIGHTS_CORRUPTED.items():
        create_corrupted(percent / 100, filename)
