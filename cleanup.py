import shutil
import os


if __name__ == "__main__":
    for dir_ in ["data", "checkpoints", "result"]:
        if os.path.exists(dir_):
            shutil.rmtree(dir_)
