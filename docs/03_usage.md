# 03 - Usage
(From StyleGAN3 readme: )

To use stylegan3 within an interactive UI you can run: ```python visualizer.py```  
To use stylegan3 within an CLI you can run:
```
# Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
python gen_images.py --outdir=out --trunc=1 --seeds=2 \
   --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

# Render a 4x2 grid of interpolations for seeds 0 through 31.
python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```
## Possible Flags
| Flag          | Image | Video | Required | Type                       | Default | Description                                                                                                                       |
|---------------|-------|-------|----------|----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| network       | yes   | yes   | yes      | URL                        | none    | network pickle filename                                                                                                           | 
| seeds         | yes   | yes   | yes      | range of integer           | none    | list of random seeds                                                                                                              | 
| trunc         | yes   | yes   | no       | float                      | 1       | truncation psi                                                                                                                    | 
| outdir        | yes   | no    | yes      | string                     | none    | where to save the output images                                                                                                   | 
| class         | yes   | no    | no       | int                        | none    | class label                                                                                                                       | 
| noise-mode    | yes   | no    | no       | ['const, 'random', 'none'] | 'const' | noise mode                                                                                                                        | 
| translate     | yes   | no    | no       | vector_float               | '0,0'   | translate XY-coordinate e.g. '0.3,1'                                                                                              | 
| rotate        | yes   | no    | no       | float                      | 0       | rotation angle in degrees                                                                                                         | 
| output        | no    | yes   | yes      | string                     | none    | output .mp4 filename                                                                                                              | 
| shuffle-seed  | no    | yes   | no       | int                        | none    | seed to use for shuffling seed order                                                                                              | 
| grid          | no    | yes   | no       | tuple                      | (1,1)   | grid width/height e.g. '4x3'                                                                                                      | 
| num-keyframes | no    | yes   | no       | int                        | none    | number of seeds to interpolate through.<br/> If not specified, deteremine based on the length of the seeds array given by --seeds | 
| w-frames      | no    | yes   | no       | int                        | 120     | number of frames to interpolate between latents                                                                                   | 
