"""
MSz Python OpenMP Example

This example demonstrates how to use MSz with OpenMP acceleration in Python.
It shows:
1. Counting topology faults with OpenMP
2. Deriving topology-preserving edits with OpenMP
3. Applying edits with OpenMP
4. Extracting critical points with OpenMP
5. Comparing performance with different thread counts
"""

import msz
import numpy as np
import os
import time

def main():
    # Use the same datasets as the C++ examples
    W, H, D = 100, 100, 1
    num_elements = W * H * D
    
    # Path to datasets relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(script_dir, "..", "datasets", "grid100x100.bin")
    decp_path = os.path.join(script_dir, "..", "datasets", "decp_grid100x100_sz3_rel_1e-4.bin")

    if not os.path.exists(orig_path) or not os.path.exists(decp_path):
        print(f"Dataset files not found at {orig_path}.")
        print("Generating dummy data for testing...")
        # Generate dummy data
        original = np.sin(np.arange(num_elements) * 0.1)
        decompressed = original + np.random.randn(num_elements) * 0.0001
    else:
        # Load data as flat float64 (double) arrays
        original = np.fromfile(orig_path, dtype=np.float64)
        decompressed = np.fromfile(decp_path, dtype=np.float64)

    if original.size != num_elements or decompressed.size != num_elements:
        print(f"Error: Array size mismatch. Expected {num_elements} elements.")
        return

    print(f"MSz Python OpenMP Example")
    print(f"Data dimensions: {W}x{H}x{D}")
    print(f"=" * 60)

    # Test different thread counts
    thread_counts = [1, 2, 4]
    
    for num_threads in thread_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_threads} OpenMP thread(s)")
        print(f"{'='*60}")
        
        # 1. Count faults with OpenMP
        print(f"\n1. Counting faults with {num_threads} thread(s)...")
        start_time = time.time()
        faults = msz.count_faults(
            original, 
            decompressed, 
            connectivity_type=0, 
            W=W, H=H, D=D,
            accelerator=msz.ACCELERATOR_OMP,
            device_id=0,
            num_omp_threads=num_threads
        )
        elapsed = time.time() - start_time
        
        if faults['status'] == msz.ERR_NO_ERROR:
            print(f"   Status: SUCCESS")
            print(f"   False minima: {faults['num_false_min']}")
            print(f"   False maxima: {faults['num_false_max']}")
            print(f"   False labels: {faults['num_false_labels']}")
            ratio = msz.calculate_false_label_ratio(
                faults['num_false_labels'], W, H, D
            )
            print(f"   False label ratio: {ratio:.6f}")
            print(f"   Time: {elapsed:.4f} seconds")
        elif faults['status'] == msz.ERR_NOT_IMPLEMENTED:
            print(f"   OpenMP not enabled in this build.")
            break
        else:
            print(f"   Error: {faults['status']}")
            continue

        # 2. Derive edits with OpenMP
        print(f"\n2. Deriving edits with {num_threads} thread(s)...")
        start_time = time.time()
        status, edits = msz.derive_edits(
            original,
            decompressed,
            preservation_options=msz.PRESERVE_MIN | msz.PRESERVE_MAX,
            connectivity_type=0,
            W=W, H=H, D=D,
            rel_err_bound=1e-4,
            accelerator=msz.ACCELERATOR_OMP,
            device_id=0,
            num_omp_threads=num_threads
        )
        elapsed = time.time() - start_time
        
        if status == msz.ERR_NO_ERROR:
            print(f"   Status: SUCCESS")
            print(f"   Derived {len(edits)} edits")
            if len(edits) > 0:
                print(f"   First edit: index={edits[0].index}, offset={edits[0].offset:.6e}")
            print(f"   Time: {elapsed:.4f} seconds")
        elif status == msz.ERR_NOT_IMPLEMENTED:
            print(f"   OpenMP not enabled in this build.")
            break
        else:
            print(f"   Error: {status}")
            continue

        # 3. Apply edits with OpenMP
        if len(edits) > 0:
            print(f"\n3. Applying edits with {num_threads} thread(s)...")
            edited_data = decompressed.copy()
            start_time = time.time()
            status_apply = msz.apply_edits(
                edited_data,
                edits,
                W=W, H=H, D=D,
                accelerator=msz.ACCELERATOR_OMP,
                device_id=0,
                num_omp_threads=num_threads
            )
            elapsed = time.time() - start_time
            
            if status_apply == msz.ERR_NO_ERROR:
                print(f"   Status: SUCCESS")
                print(f"   Time: {elapsed:.4f} seconds")
                
                # Verify faults after edits
                faults_after = msz.count_faults(
                    original, 
                    edited_data, 
                    connectivity_type=0, 
                    W=W, H=H, D=D,
                    accelerator=msz.ACCELERATOR_OMP,
                    num_omp_threads=num_threads
                )
                print(f"   Faults after edits:")
                print(f"     False minima: {faults_after['num_false_min']}")
                print(f"     False maxima: {faults_after['num_false_max']}")
                print(f"     False labels: {faults_after['num_false_labels']}")
            else:
                print(f"   Error: {status_apply}")

        # 4. Extract critical points with OpenMP
        print(f"\n4. Extracting critical points with {num_threads} thread(s)...")
        start_time = time.time()
        result = msz.extract_critical_points(
            original,
            connectivity_type=0,
            W=W, H=H, D=D,
            accelerator=msz.ACCELERATOR_OMP,
            device_id=0,
            num_omp_threads=num_threads
        )
        elapsed = time.time() - start_time
        
        if result['status'] == msz.ERR_NO_ERROR:
            print(f"   Status: SUCCESS")
            print(f"   Found {len(result['minima'])} minima")
            print(f"   Found {len(result['maxima'])} maxima")
            if len(result['minima']) > 0:
                cp = result['minima'][0]
                print(f"   First minimum: pos=({cp.x},{cp.y},{cp.z}), value={cp.value:.6f}")
            print(f"   Time: {elapsed:.4f} seconds")
        else:
            print(f"   Error: {result['status']}")

    # 5. Demonstrate Zstd compression (works independently of accelerator)
    if 'edits' in locals() and len(edits) > 0:
        print(f"\n{'='*60}")
        print("5. Testing Zstd compression of edits")
        print(f"{'='*60}")
        status_comp, compressed = msz.compress_edits_zstd(edits)
        if status_comp == msz.ERR_NO_ERROR:
            print(f"   Original edits: {len(edits)}")
            print(f"   Compressed size: {len(compressed)} bytes")
            print(f"   Compression ratio: {(len(edits) * 12) / len(compressed):.2f}x")
            
            # Decompress and verify
            status_dec, edits_dec = msz.decompress_edits_zstd(compressed)
            if status_dec == msz.ERR_NO_ERROR:
                print(f"   Decompressed edits: {len(edits_dec)}")
                print(f"   Verification: {'PASSED' if len(edits_dec) == len(edits) else 'FAILED'}")
        else:
            print(f"   Zstd compression not available (Error {status_comp})")

    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
