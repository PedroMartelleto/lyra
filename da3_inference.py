import torch
from depth_anything_3.api import DepthAnything3

from decord import VideoReader, cpu
from time import time

images_list = []

for i in range(6):
    input_video = f"assets/demo/static/diffusion_output_generated/{i}/rgb/Garden.mp4"
    print(f"Loading video: {input_video}")
    vr = VideoReader(input_video, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frames = vr.get_batch(range(len(vr))).asnumpy()
    H, W = frames.shape[1:3]

    for i in range(len(frames)):
        images_list.append(frames[i])

print("Total frames before subsampling:", len(images_list))

images_list = images_list[::10]

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = images_list
print(f"Found {len(images)} images.")

start = time()

export_kwargs = {
    "gs_video": {
        "chunk_size": 1,
        "video_quality": "high"
    }
}

prediction = model.inference(
    images,
    infer_gs=True,
    export_dir="output_render_da3_3",
    process_res=1024,
    num_max_points=500_000,
    export_format="gs_video-gs_ply",
    export_kwargs=export_kwargs
)

print(f"Total inference time: {time() - start} seconds")
