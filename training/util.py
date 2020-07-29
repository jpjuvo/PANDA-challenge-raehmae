import os
import shutil

def makedir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def move_models_2_savedir(SAVE_DIR, models=None):
    makedir(SAVE_DIR)
    for model in models:
        basename = os.path.basename(model)
        shutil.move(model, os.path.join(SAVE_DIR, basename))