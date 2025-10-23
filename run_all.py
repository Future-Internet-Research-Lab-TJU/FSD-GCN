import os
import subprocess

folder = "./export_lp/test"
for file in os.listdir(folder):
    if file.endswith('.py'):
        print(f"{file}...")

        subprocess.run(["python", os.path.join(folder, file)])
