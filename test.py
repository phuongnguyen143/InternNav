from pxr import Usd

# 1. Load the USD scene stage
usd_file_path = "data/Embodiments/vln-pe/h1/h1_internvla.usd"
stage = Usd.Stage.Open(usd_file_path)
print(stage)
# 2. Traverse and print all Prims (Objects) in the scene hierarchy
for prim in stage.Traverse():
    print(f"Prim Name: {prim.GetName()} | Type: {prim.GetTypeName()}")
