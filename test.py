import copy
import glob
import os
import shutil
import sys
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model import unet


def messenger(func):
    def wrapper(*args, **kwargs):
        print('--start--')
        func(*args, **kwargs)
        print('--end--')
    return wrapper


@messenger
def main():
    test_images = glob.glob(os.path.join("samples", "test", "image", "*.jpg"))

    if not test_images:
        print("No image in samples/test/image/*.jpg")
        sys.exit()

    if os.path.exists("result"):
        shutil.rmtree("result")
    os.makedirs("result")

    for test_image in test_images:
        img_name = os.path.basename(test_image)
        image = Image.open(test_image)
        try:
            if len(image.getdata()[0]) != 1:
                print("グレスケ変換します")
                image = image.convert('L')
        except:
            pass

        print("Processing: ", img_name)
        resized_image = sizeChecker(image)
        predicted_image = inference(resized_image)

        predicted_image.save(os.path.join("result", img_name), quality=95)
        print("Predicted image saved")
        print()


def getNearestValue(num):
    list_256 = [256 * i for i in range(20)]
    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list_256) - num).argmin()
    return list_256[idx]


def sizeChecker(image):
    w = image.size[0]
    h = image.size[1]

    if (w % 256 != 0) or (h % 256 != 0):
        print("Warning: Inappropriate　image size", w, h)
        w = getNearestValue(w)
        h = getNearestValue(h)
        print("Input image resized to ", w, h)
        image = image.resize((w, h), Image.LANCZOS)
    else:
        print("Input image size:", w, h)

    return image


def cut_256(image):
    w = image.size[0]
    h = image.size[1]

    image = np.array(image)

    h_list = range(0, h, 256)
    w_list = range(0, w, 256)

    images_256 = []
    for i in h_list:
        for j in w_list:
            temp = image[i:i+256, j:j+256]
            images_256.append(copy.deepcopy(temp))

    return images_256


def inference(image):
    w = image.size[0]
    h = image.size[1]

    num_w = w//256
    num_h = h//256

    images_256 = cut_256(image)

    model = unet()
    model.load_weights(os.path.join("checkpoints", "unet.hdf5"))

    images_pred = []
    for image in images_256:
        image_pred = model.predict(image.reshape(256, 256, 1).reshape(1,
                                                                      256, 256,
                                                                      1))
        images_pred.append(image_pred)

    predicted_image = reincarnation(images_pred, num_w, num_h)

    return predicted_image


def reincarnation(images_pred, num_w, num_h):
    images_pred = [pred[0, :, :, 0] for pred in images_pred]
    lines = []

    for i in range(num_h):
        line = np.hstack(images_pred[num_w*i:num_w*(i+1)])
        lines.append(line)

    image = np.vstack(lines)
    image = image*255
    # image[image<150] = 0
    # image[image>=150] = 255

    predicted_image = Image.fromarray(np.uint8(image))
    return predicted_image


if __name__ == "__main__":
    main()
