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

    # 1. Count faults
    faults = msz.count_faults(
        original, 
        decompressed, 
        connectivity_type=0, 
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    print("Initial Faults count:", faults)

    # 2. Derive edits
    # Match C++: MSZ_PRESERVE_MIN | MSZ_PRESERVE_MAX, connectivity=0, rel_err=1e-4
    status, edits = msz.derive_edits(
        original,
        decompressed,
        preservation_options=msz.PRESERVE_MIN | msz.PRESERVE_MAX,
        connectivity_type=0,
        W=W, H=H, D=D,
        rel_err_bound=1e-4,
        accelerator=msz.ACCELERATOR_NONE
    )

    if status == msz.ERR_NO_ERROR:
        print(f"Derived {len(edits)} edits.")
        if len(edits) > 0:
            print("First edit:", edits[0])
    else:
        print(f"Error deriving edits: {status}")
        return

    # 3. Apply edits
    # Apply to a copy to keep original decompressed data intact
    edited_data = decompressed.copy()
    status_apply = msz.apply_edits(
        edited_data,
        edits,
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    print(f"Apply edits status: {status_apply}")

    # 4. Verify faults after edits
    faults_after = msz.count_faults(
        original, 
        edited_data, 
        connectivity_type=0, 
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    print("Faults count after edits:", faults_after)

    # 5. Optional: Zstd Compression (if enabled)
    status_comp, compressed = msz.compress_edits_zstd(edits)
    if status_comp == msz.ERR_NO_ERROR:
        print(f"Compressed edits size: {len(compressed)} bytes")
        status_dec, edits_dec = msz.decompress_edits_zstd(compressed)
        print(f"Decompressed {len(edits_dec)} edits.")
    else:
        print(f"Zstd Compression not available (Error {status_comp})")

if __name__ == "__main__":
    main()
