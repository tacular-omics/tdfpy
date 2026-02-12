import time
from tdfpy import timsdata

TDF_PATH = r"tests/data/200ngHeLaPASEF_1min.d"


def test_connection_overhead():
    """Compare overhead of reusing vs reopening connections."""
    
    # Get a frame ID to test with
    with timsdata.timsdata_connect(TDF_PATH) as td:
        cursor = td.conn.cursor()
        cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 1")
        frame_id = cursor.fetchone()[0]
    
    NUM_READS = 1000
    
    # Test 1: Reuse connection
    start = time.perf_counter()
    with timsdata.timsdata_connect(TDF_PATH) as td:
        for _ in range(NUM_READS):
            mz = td.readScans(frame_id, 0, 500)
    reuse_time = time.perf_counter() - start
    
    # Test 2: Open new connection each time
    start = time.perf_counter()
    for _ in range(NUM_READS):
        with timsdata.timsdata_connect(TDF_PATH) as td:
            mz = td.readScans(frame_id, 0, 500)
    reopen_time = time.perf_counter() - start
    
    print(f"\nConnection Overhead Test ({NUM_READS} reads):")
    print(f"Reuse connection:  {reuse_time:.4f}s ({reuse_time/NUM_READS*1000:.2f}ms per read)")
    print(f"Reopen each time:  {reopen_time:.4f}s ({reopen_time/NUM_READS*1000:.2f}ms per read)")
    print(f"Overhead factor:   {reopen_time/reuse_time:.1f}x slower")
    print(f"Extra cost:        {(reopen_time-reuse_time)/NUM_READS*1000:.2f}ms per connection")


if __name__ == "__main__":
    test_connection_overhead()