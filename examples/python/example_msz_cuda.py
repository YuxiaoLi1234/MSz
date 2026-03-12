import msz
import numpy as np
import os
import time

def main():
    W, H, D = 150, 450, 1
    num_elements = W * H * D

    # Path to datasets relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(script_dir, "..", "datasets", "heated.bin")
    decp_path = os.path.join(script_dir, "..", "datasets", "decp_heated_1e-3_150_450.bin")

    if not os.path.exists(orig_path) or not os.path.exists(decp_path):
        print(f"Dataset files not found. Generating dummy data for testing...")
        np.random.seed(42)
        original = np.sin(np.linspace(0, 10 * np.pi, num_elements)) + 0.1 * np.random.randn(num_elements)
        decompressed = original + np.random.randn(num_elements) * 1e-3
    else:
        original = np.fromfile(orig_path, dtype=np.float64)
        decompressed = np.fromfile(decp_path, dtype=np.float64)

    if original.size != num_elements or decompressed.size != num_elements:
        print(f"Error: Array size mismatch. Expected {num_elements}, got orig={original.size}, decp={decompressed.size}")
        return

    print(f"MSz Python CUDA vs Serial Comparison")
    print(f"Data dimensions: {W}x{H}x{D} ({num_elements} elements)")
    print(f"{'='*60}")

    rel_err_bound = 1e-3
    connectivity_type = 0
    preservation_options = msz.PRESERVE_MIN | msz.PRESERVE_MAX

    # --- Count faults before fixing ---
    print("\n1. Counting faults in decompressed data (before fix)...")
    faults_before = msz.count_faults(
        original, decompressed,
        connectivity_type=connectivity_type,
        W=W, H=H, D=D,
        accelerator=msz.ACCELERATOR_NONE
    )
    print(f"   False minima:  {faults_before['num_false_min']}")
    print(f"   False maxima:  {faults_before['num_false_max']}")
    print(f"   False labels:  {faults_before['num_false_labels']}")

    # --- Serial (CPU) ---
    print(f"\n2. Deriving edits (Serial / CPU)...")
    t0 = time.time()
    status_cpu, edits_cpu = msz.derive_edits(
        original, decompressed,
        preservation_options=preservation_options,
        connectivity_type=connectivity_type,
        W=W, H=H, D=D,
        rel_err_bound=rel_err_bound,
        accelerator=msz.ACCELERATOR_NONE
    )
    elapsed_cpu = time.time() - t0

    if status_cpu == msz.ERR_NO_ERROR:
        print(f"   Status:  SUCCESS")
        print(f"   Edits:   {len(edits_cpu)}")
        print(f"   Time:    {elapsed_cpu:.4f}s")
    else:
        print(f"   ERROR: status={status_cpu}")
        return

    # --- CUDA ---
    print(f"\n3. Deriving edits (CUDA)...")
    t0 = time.time()
    status_gpu, edits_gpu = msz.derive_edits(
        original, decompressed,
        preservation_options=preservation_options,
        connectivity_type=connectivity_type,
        W=W, H=H, D=D,
        rel_err_bound=rel_err_bound,
        accelerator=msz.ACCELERATOR_CUDA, device_id=0
    )
    elapsed_gpu = time.time() - t0

    if status_gpu == msz.ERR_NO_ERROR:
        print(f"   Status:  SUCCESS")
        print(f"   Edits:   {len(edits_gpu)}")
        print(f"   Time:    {elapsed_gpu:.4f}s")
    elif status_gpu == msz.ERR_NOT_IMPLEMENTED:
        print(f"   CUDA not enabled in this build.")
        return
    else:
        print(f"   ERROR: status={status_gpu}")
        return

    # --- Comparison ---
    print(f"\n{'='*60}")
    print(f"Comparison: CPU vs CUDA")
    print(f"{'='*60}")
    print(f"   CPU  edits: {len(edits_cpu)}")
    print(f"   CUDA edits: {len(edits_gpu)}")
    print(f"   Match:      {'YES ✓' if len(edits_cpu) == len(edits_gpu) else 'NO ✗ (count differs)'}")
    if elapsed_cpu > 0:
        print(f"   Speedup:    {elapsed_cpu / elapsed_gpu:.2f}x")

    # --- Verify faults after applying CPU edits ---
    print(f"\n4. Verifying faults after applying CPU edits...")
    edited_cpu = decompressed.copy()
    msz.apply_edits(edited_cpu, edits_cpu, W=W, H=H, D=D, accelerator=msz.ACCELERATOR_NONE)
    faults_cpu = msz.count_faults(original, edited_cpu, connectivity_type=connectivity_type, W=W, H=H, D=D, accelerator=msz.ACCELERATOR_NONE)
    print(f"   False minima:  {faults_cpu['num_false_min']}")
    print(f"   False maxima:  {faults_cpu['num_false_max']}")
    print(f"   False labels:  {faults_cpu['num_false_labels']}")

    # --- Verify faults after applying CUDA edits ---
    print(f"\n5. Verifying faults after applying CUDA edits...")
    edited_gpu = decompressed.copy()
    status = msz.apply_edits(edited_gpu, edits_gpu, W=W, H=H, D=D, accelerator=msz.ACCELERATOR_CUDA)
    print(f"   Status: {status}")
    faults_gpu = msz.count_faults(original, edited_gpu, connectivity_type=connectivity_type, W=W, H=H, D=D, accelerator=msz.ACCELERATOR_CUDA)
    print(f"   False minima:  {faults_gpu['num_false_min']}")
    print(f"   False maxima:  {faults_gpu['num_false_max']}")
    print(f"   False labels:  {faults_gpu['num_false_labels']}")

    print(f"\n{'='*60}")
    ok = (faults_cpu['num_false_min'] == faults_gpu['num_false_min'] and
          faults_cpu['num_false_max'] == faults_gpu['num_false_max'] and
          faults_cpu['num_false_labels'] == faults_gpu['num_false_labels'])
    print(f"Overall result: {'PASSED ✓' if ok else 'FAILED ✗ — CUDA results differ from CPU'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
