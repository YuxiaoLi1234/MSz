import msz
import numpy as np
import os

def main():
    # Use the same datasets as the C++ examples
    W, H, D = 100, 100, 1
    num_elements = W * H * D
    
    # Path to datasets relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(script_dir, "..", "datasets", "grid100x100.bin")
    decp_path = os.path.join(script_dir, "..", "datasets", "decp_grid100x100_sz3_rel_1e-4.bin")

    if not os.path.exists(orig_path) or not os.path.exists(decp_path):
        print(f"Dataset files not found at {orig_path}. Please run from the project root.")
        return
    
    # Load data as flat float64 (double) arrays to match C++ vector<double>
    original = np.fromfile(orig_path, dtype=np.float64)
    decompressed = np.fromfile(decp_path, dtype=np.float64)

    if original.size != num_elements or decompressed.size != num_elements:
        print(f"Error: File size mismatch. Expected {num_elements} elements.")
        return

    print(f"Data dimensions: {W}x{H}x{D}")

    # 1. Extract critical points from original data
    print("\n" + "="*60)
    print("1. Extracting critical points from ORIGINAL data...")
    print("="*60)
    
    result = msz.extract_critical_points(
        original,
        connectivity_type=0,
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    
    if result['status'] == msz.ERR_NO_ERROR:
        minima = result['minima']
        maxima = result['maxima']
        saddles = result['saddles']
        
        print(f"Found {len(minima)} minima, {len(maxima)} maxima, and {len(saddles)} saddle points.")
        
        # Display first few minima
        print("\n=== Minima (first 5) ===")
        for i, cp in enumerate(minima[:5]):
            print(f"  [{i}] Position=({cp.x}, {cp.y}, {cp.z}), Value={cp.value:.6e}, Index={cp.index}")
        if len(minima) > 5:
            print(f"  ... and {len(minima) - 5} more.")
        
        # Display first few maxima
        print("\n=== Maxima (first 5) ===")
        for i, cp in enumerate(maxima[:5]):
            print(f"  [{i}] Position=({cp.x}, {cp.y}, {cp.z}), Value={cp.value:.6e}, Index={cp.index}")
        if len(maxima) > 5:
            print(f"  ... and {len(maxima) - 5} more.")
        
        # Display first few saddle points
        print("\n=== Saddle Points (first 5) ===")
        for i, cp in enumerate(saddles[:5]):
            print(f"  [{i}] Position=({cp.x}, {cp.y}, {cp.z}), Value={cp.value:.6e}, Index={cp.index}")
        if len(saddles) > 5:
            print(f"  ... and {len(saddles) - 5} more.")
    else:
        print(f"Error extracting critical points: {result['status']}")

    # 2. Count faults
    print("\n" + "="*60)
    print("2. Counting faults in decompressed data...")
    print("="*60)
    faults = msz.count_faults(
        original, 
        decompressed, 
        connectivity_type=0, 
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    print("Initial Faults count:", faults)

    # 3. Derive edits
    print("\n" + "="*60)
    print("3. Deriving edits to fix faults...")
    print("="*60)
    # Match C++: MSZ_PRESERVE_MIN | MSZ_PRESERVE_MAX, connectivity=0, rel_err=1e-4
    status, edits = msz.derive_edits(
        original,
        decompressed,
        preservation_options=msz.PRESERVE_MIN | msz.PRESERVE_MAX,
        connectivity_type=0,
        W=W, H=H, D=D,
        rel_err_bound=0.1,
        accelerator=msz.ACCELERATOR_NONE
    )

    if status == msz.ERR_NO_ERROR:
        print(f"Derived {len(edits)} edits.")
        if len(edits) > 0:
            print("First edit:", edits[0])
    else:
        print(f"Error deriving edits: {status}")
        return

    # 4. Apply edits
    print("\n" + "="*60)
    print("4. Applying edits...")
    print("="*60)
    # Apply to a copy to keep original decompressed data intact
    edited_data = decompressed.copy()
    status_apply = msz.apply_edits(
        edited_data,
        edits,
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    print(f"Apply edits status: {status_apply}")

    # 5. Verify faults after edits
    faults_after = msz.count_faults(
        original, 
        edited_data, 
        connectivity_type=0, 
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    print("Faults count after edits:", faults_after)

    # 6. Optional: Zstd Compression (if enabled)
    print("\n" + "="*60)
    print("5. Testing Zstd compression...")
    print("="*60)
    status_comp, compressed = msz.compress_edits_zstd(edits)
    if status_comp == msz.ERR_NO_ERROR:
        print(f"Compressed edits size: {len(compressed)} bytes")
        status_dec, edits_dec = msz.decompress_edits_zstd(compressed)
        print(f"Decompressed {len(edits_dec)} edits.")
    else:
        print(f"Zstd Compression not available (Error {status_comp})")

if __name__ == "__main__":
    main()

