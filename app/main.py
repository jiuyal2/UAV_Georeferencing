from georef import generate
import os

print(os.path.exists("./assets/153/153_vid_destb.mp4"))

generate(R"./assets/153/153_vid_destb.mp4", R"./assets/153/153_tgt.tif")