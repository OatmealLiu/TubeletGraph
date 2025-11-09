if [ ! -d "./_ckpts" ]; then
    mkdir ./_ckpts
fi

if [ ! -f "./_ckpts/CropFormer_hornet_3x_03823a.pth" ]; then
    python -m gdown 1b36D3PEh2TqzqXBCDhTySHmla2MltXVU -O ./_ckpts/CropFormer_hornet_3x_03823a.pth
fi
