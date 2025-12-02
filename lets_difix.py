import os
import sys
import math
import torch
import torch.multiprocessing as mp
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import imageio

current_dir = os.getcwd()
module_path = os.path.join(current_dir, "Difix3D", "src")
if module_path not in sys.path:
    sys.path.append(module_path)

from pipeline_difix import DifixPipeline

def process_video_shard(gpu_id, video_tasks, ref_video_path):
    """
    Worker function to process a list of videos on a specific GPU.
    """
    device = f"cuda:{gpu_id}"
    print(f"[{device}] Initializing model...")

    # Load model
    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix_ref", 
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    pipe.to(device)

    # Load Reference Image Source (Frame 0 of reference video)
    print(f"[{device}] Loading reference frame source...")
    try:
        vr_ref = VideoReader(ref_video_path, ctx=cpu(0))
        ref_frame_np = vr_ref[0].asnumpy()
        source_ref_image = Image.fromarray(ref_frame_np)
    except Exception as e:
        print(f"[{device}] Error loading reference video {ref_video_path}: {e}")
        return

    for input_path, output_path in video_tasks:
        print(f"[{device}] Processing: {input_path} -> {output_path} (Side-by-Side)")
        
        if not os.path.exists(input_path):
            print(f"[{device}] Warning: Input file not found: {input_path}")
            continue

        try:
            # Load Input Video
            vr = VideoReader(input_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            
            # Get input dimensions
            h, w, _ = vr[0].shape 
            
            # Resize reference image to match input video dimensions
            if source_ref_image.size != (w, h):
                ref_image_resized = source_ref_image.resize((w, h), Image.LANCZOS)
            else:
                ref_image_resized = source_ref_image

            # Setup Video Writer
            # Note: The output width will be 2 * w, imageio handles this automatically
            # on the first frame written.
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

            # Process frame by frame
            for i in range(total_frames):
                # Raw numpy array (Before)
                frame_np = vr[i].asnumpy()
                input_image = Image.fromarray(frame_np)

                with torch.no_grad():
                    output_image_pil = pipe(
                        prompt="remove degradation",
                        image=input_image,
                        ref_image=ref_image_resized,
                        num_inference_steps=1,
                        timesteps=[199],
                        guidance_scale=0.0
                    ).images[0]

                # Convert output to numpy (After)
                output_np = np.array(output_image_pil)
                
                # Create Side-by-Side: [Before | After]
                # Concatenate along width (axis 1)

                writer.append_data(output_np)

                if i % 10 == 0:
                    print(f"[{device}] Video {os.path.basename(input_path)}: {i}/{total_frames}")

            writer.close()
            print(f"[{device}] Finished {output_path}")
            
        except Exception as e:
            print(f"[{device}] Error processing {input_path}: {e}")
            import traceback
            traceback.print_exc()

def main():
    # --- Configuration ---
    num_gpus = 4
    base_input_dir = "outputs/demo/lyra_dynamic_3/static_view_indices_fixed_5_0_1_2_3_4/lyra_dynamic_demo_generated/0/raw"
    output_dir = "outputs/difix_3"
    ref_video_path = "test_inference2/rgb/815e2628-544a-50ea-b746-bef4b9e9b695.mp4"
    
    os.makedirs(output_dir, exist_ok=True)

    views_indices = range(6) # 0 to 5
    
    tasks = []
    for idx in views_indices:
        input_filename = f"rgb_0_view_idx_{idx}.mp4"
        input_path = os.path.join(base_input_dir, input_filename)
        output_filename = f"view_{idx}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        tasks.append((input_path, output_path))

    # --- Distribute Work ---
    gpu_tasks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        gpu_idx = i % num_gpus
        gpu_tasks[gpu_idx].append(task)

    processes = []
    mp.set_start_method('spawn', force=True)

    print(f"Starting processing on {num_gpus} GPUs...")
    
    for gpu_id in range(num_gpus):
        if not gpu_tasks[gpu_id]:
            continue 
            
        p = mp.Process(
            target=process_video_shard, 
            args=(gpu_id, gpu_tasks[gpu_id], ref_video_path)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processing complete.")

if __name__ == "__main__":
    main()