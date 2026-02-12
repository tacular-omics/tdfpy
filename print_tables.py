import tdfpy as tdf


with tdf.DDA("tests/data/200ngHeLaPASEF_1min.d") as dda:
    tables = dda.pandas_tdf

    print(tables.precursors)
    print(tables.frames)
    print(tables.pasef_frame_msms_info)