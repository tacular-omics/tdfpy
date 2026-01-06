import tdfpy as tp

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    with tp.timsdata_connect("/home/patrick-garrett/Downloads/HeLa_5min_raw/20180924_50ngHeLa_1.0.25.1_Hystar5.0SR1_S2-B4_1_2057.d") as td:
        print(td)

        for spectra in tp.get_centroided_ms1_spectra(td, num_workers=1, ordered=False):
            print(spectra)


if __name__ == "__main__":
    main()