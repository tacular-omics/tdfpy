"""Profile the centroiding performance of TDFpy."""
import cProfile
import pstats
import io
from pstats import SortKey

import tdfpy as tp


def profile_centroiding():
    """Profile centroiding on test data."""
    with tp.timsdata_connect(
        "/home/patrick-garrett/Repos/tdfpy/tests/data/200ngHeLaPASEF_1min.d"
    ) as td:
        # Get first 5 MS1 frames for profiling
        cursor = td.conn.cursor()
        cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 5")
        frame_ids = [row[0] for row in cursor.fetchall()]
        
        print(f"Profiling centroiding on {len(frame_ids)} frames...")
        
        # Profile the centroiding
        for spectrum in tp.get_centroided_ms1_spectra(td, frame_ids=frame_ids):
            pass  # Just iterate, don't process


if __name__ == "__main__":
    # Create profiler
    profiler = cProfile.Profile()
    
    # Run profiling
    profiler.enable()
    profile_centroiding()
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("TOP TIME-CONSUMING FUNCTIONS")
    print("="*80)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.TIME)
    ps.print_stats(20)  # Top 20 by time
    print(s.getvalue())
