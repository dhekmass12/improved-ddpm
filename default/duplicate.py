import shutil

for i in range(63):
    shutil.copy("default/samples/x0_0.png", f"default/samples/x0_{i+1}.png")