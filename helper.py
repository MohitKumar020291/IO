import os
import yaml
import hashlib

def generateFilename(
    base_name: str = None, 
    string: str = None, 
    run_num: int = 1,
    ext: str = None
) -> str:
    if ext and not isinstance(ext, str):
        raise ValueError(f"extension should be of type str, got {type(ext)}")
    if base_name:
        base_name, ext = os.path.splitext(base_name)
    if ext == "" or ext == None: #pain point
        raise Exception("please provide an extension")
    if not base_name:
        hash_filename = string or "filename"
        sha256_hash = hashlib.sha256()
        sha256_hash.update(hash_filename.encode())
        hex_digest = sha256_hash.hexdigest()
        if os.path.exists(hex_digest + ext):
            return generateFilename(base_name=hex_digest, ext=ext)
        else:
            return hex_digest + ext
    else:
        if os.path.exists(base_name + ext):
            base_name = base_name + str(run_num)
            if os.path.exists(base_name + ext):
                return generateFilename(base_name=base_name, run_num=run_num+1, ext=ext)
            else:
                return base_name + ext
        else:
            return base_name + ext

def read_yaml(
    file_path: str
) -> dict:
    assert os.path.exists(file_path), f"{file_path} does not exist!"
    base_name, ext = os.path.splitext(file_path)
    assert ext in [".yaml", ".yml"], "file_path should have a yaml or yml extension."
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        raise e
    return data
