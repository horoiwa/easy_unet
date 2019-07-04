import os
from PIL import Image
import numpy as np
import glob
import shutil


def prep_data(n_sample=30):

    path_images = glob.glob(os.path.join("samples", "train", "image", "*.jpg"))
    path_masks = glob.glob(os.path.join("samples", "train", "mask", "*.jpg"))
    dataset = zip(path_images, path_masks)

    if os.path.exists("data"):
        print("data dir が存在するので初期化します")
        shutil.rmtree("data")

    os.makedirs(os.path.join("data", "image"))
    os.makedirs(os.path.join("data", "mask"))

    path_saveimage = os.path.join("data", "image", "")
    path_savemask = os.path.join("data", "mask", "")

    idx = 1
    for (image, mask) in dataset:
        assert os.path.basename(image) == os.path.basename(mask), "ファイル名不一致"
        n = 0
        while n < n_sample:
            image_sample, mask_sample = sampling256(image, mask)

            image_sample.save(path_saveimage+str(idx)+".jpg", quality=95)
            mask_sample.save(path_savemask+str(idx)+".jpg", quality=95)
            n += 1
            idx += 1

    print("Sampling finished gracefully")


def sampling256(image, mask):
    image = Image.open(image)
    mask = Image.open(mask)
    assert image.size == mask.size, "元画像とマスク画像のアス比が違う"

    try:
        if len(image.getdata()[0]) != 1:
            print("グレースケール変換します")
            image = image.convert('L')
            mask = mask.convert('L')
    except:
        pass

    image = np.array(image)
    mask = np.array(mask)

    h = image.shape[0]
    w = image.shape[1]

    assert h >= 256, "ImageSizeError: 縦が小さすぎ"
    assert w >= 256, "ImageSizeError: 横が小さすぎ"

    random_h = np.random.randint(0, h-256+1)
    random_w = np.random.randint(0, w-256+1)

    image_sample = image[random_h:random_h+256, random_w:random_w+256]
    mask_sample = mask[random_h:random_h+256, random_w:random_w+256]
    assert image_sample.shape[:2] == (256, 256), "予期せぬエラー: tag1"

    image_sample = Image.fromarray(np.uint8(image_sample))
    mask_sample = Image.fromarray(np.uint8(mask_sample))

    return image_sample, mask_sample


if __name__ == '__main__':
    prep_data(n_sample=10)
