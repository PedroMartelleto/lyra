import torch
from depth_anything_3.api import DepthAnything3

from decord import VideoReader, cpu

VIDEO_QUALITY_MAP = {
    "low": {"crf": "28", "preset": "veryfast"},
    "medium": {"crf": "23", "preset": "medium"},
    "high": {"crf": "18", "preset": "slow"},
}

input_video = "assets/demo/dynamic/diffusion_output_generated/0/rgb/781d986f-791d-52f9-801f-34d4a616e072.mp4"

print(f"Loading video: {input_video}")
vr = VideoReader(input_video, ctx=cpu(0))
fps = vr.get_avg_fps()
frames = vr.get_batch(range(len(vr))).asnumpy()
H, W = frames.shape[1:3]
images_list = [f for f in frames]

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = images_list
print(f"Found {len(images)} images.")

extrinsics = world2cam

prediction = model.inference(
    images[::4],
    extrinsics=extrinsics,
    infer_gs=True,
    export_dir="output_render_da3",
    process_res=1024,
    num_max_points=10_000_000,
    export_format="gs_video",
    trj_mode="original",
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)

# filenames = [f"frame_{i:05d}.png" for i in range(len(images))]
# os.makedirs("output_render_da3", exist_ok=True)

# for i, depth_map in enumerate(prediction.depth):
#     # Method A: Use DA3's internal visualizer (Recommended)
#     # This handles percentile clipping and normalization automatically
#     vis = visualize_depth(depth_map)
    
#     # Save the file
#     save_path = os.path.join("output_render_da3", filenames[i])
#     imageio.imwrite(save_path, vis)
#     print(f"Saved {save_path}")