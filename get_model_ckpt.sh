#!/bin/bash

links=(
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth
https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth
)

for i in ${links[@]}; do
    wget $i
done