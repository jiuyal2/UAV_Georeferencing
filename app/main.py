from georef import generate
import os

TAG = 212

print(os.path.exists(Rf"./assets/{TAG}/{TAG}_vid_destb.mp4"))

generate(Rf"./assets/{TAG}/{TAG}_vid_destb.mp4", Rf"./assets/{TAG}/{TAG}_tgt1.tif",
         f"{TAG}_transforms.txt", vid_out_path = f"{TAG}_stb.mp4")