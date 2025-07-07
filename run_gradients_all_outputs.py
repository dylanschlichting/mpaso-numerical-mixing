import os
import subprocess
print('script started')

cwd = os.path.basename(os.getcwd())

for file in os.listdir('.'):
        if file.startswith('output_member'):
            print(f"file found {file}")
            result = subprocess.run(['python3','uv_deriv.py',f'--mesh-file={cwd}_channel_init.nc',f'--flow-file={file}'])
            if result.returncode ==0:
                print(f"{file} processed successfully")
            else:
                print(f"error processing {file}")
                print("STDERR:\n", result.stderr)
    