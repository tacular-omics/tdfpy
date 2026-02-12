from tdfpy import DIA, DDA

def main():

    dia = DIA("/home/patrick-garrett/Data/Arabela/seminal_plasma/raw/DIA/boar/20220217_Boar-1_S3-A7_1_8655.d")
    for window in dia.windows:
        print(window.centroid().shape)
        print(len(window.peaks))
        break

    for ms1 in dia.ms1_frames:
        print(ms1.centroid().shape)
        print(len(ms1.peaks))
        break

    dda = DDA("/home/patrick-garrett/Data/Arabela/seminal_plasma/raw/DDA/boar/20220217_Boar-1_S3-A7_1_8553.d")
    for precursor in dda.precursors:
        print(len(precursor.peaks))
        # no centroiding for precursors, has internal method for this
        break

    for ms1 in dda.ms1_frames:
        print(ms1.centroid().shape)
        print(len(ms1.peaks))
        break

if __name__ == "__main__":
    main()