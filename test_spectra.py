from tdfpy import DIA, DDA

def main():

    with DIA("/home/patrick-garrett/Data/Arabela/seminal_plasma/raw/DIA/boar/20220217_Boar-1_S3-A7_1_8655.d") as dia:
        for window in dia.windows:
            print(window.centroid().shape)
            print(len(window.peaks))
            break

        for ms1 in dia.ms1_frames:
            print(ms1.centroid().shape)
            print(len(ms1.peaks))
            break

    with DDA("/home/patrick-garrett/Data/Arabela/seminal_plasma/raw/DDA/boar/20220217_Boar-1_S3-A7_1_8553.d") as dda:
        for precursor in dda.precursors:
            print(len(precursor.peaks))
            break

        for ms1 in dda.ms1_frames:
            print(ms1.centroid().shape)
            print(len(ms1.peaks))
            break

if __name__ == "__main__":
    main()