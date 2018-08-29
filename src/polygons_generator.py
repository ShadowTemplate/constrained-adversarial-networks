import numpy as np
import os
import random

from experiment import load_experiment_config
from itertools import combinations, product
from PIL import Image
from shutil import make_archive
from utils import get_module_functions

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _apply_parity_check(image, img_width, img_height):
    assert img_width % 2 == 0
    assert img_height % 2 == 0

    for j in range(1, img_height - 1):  # rows
        sum_left = [image.getpixel((j, c)) for c in range(
            1, int(img_width / 2))].count(WHITE)
        sum_right = [image.getpixel((j, c)) for c in range(
            int(img_width / 2), img_width - 1)].count(WHITE)
        if sum_left % 2 == 0:
            image.putpixel((j, 0), BLACK)
        else:
            image.putpixel((j, 0), WHITE)

        if sum_right % 2 == 0:
            image.putpixel((j, img_width - 1), BLACK)
        else:
            image.putpixel((j, img_width - 1), WHITE)

    for j in range(1, img_width - 1):  # cols
        sum_top = [image.getpixel((r, j)) for r in range(
            1, int(img_height / 2))].count(WHITE)
        sum_bottom = [image.getpixel((r, j)) for r in range(
            int(img_height / 2), img_height - 1)].count(WHITE)
        if sum_top % 2 == 0:
            image.putpixel((0, j), BLACK)
        else:
            image.putpixel((0, j), WHITE)

        if sum_bottom % 2 == 0:
            image.putpixel((img_height - 1, j), BLACK)
        else:
            image.putpixel((img_height - 1, j), WHITE)


def _generate_image(img_width, img_height, area, shapes, fns, generated_images,
                    parity_check):
    # new image with black background
    image = Image.new('RGB', (img_width, img_height), BLACK)

    while True:
        polygons = [fns.get("_polygon_generator_type%s" % s)
                    (img_width, img_height, area) for s in shapes]
        signature = tuple([tuple(p) for p in sorted(polygons)])
        if signature not in generated_images and \
                not _incompatible_polygons(polygons):
            generated_images.add(signature)
            for polygon in polygons:
                for pixel in polygon:
                    image.putpixel((pixel[0], pixel[1]), WHITE)

            if parity_check:
                _apply_parity_check(image, img_width, img_height)
            break

    return image


def _get_random_shapes(dataset_size, polygons_prob):
    random_output = np.random.multinomial(dataset_size, polygons_prob)
    polygon_types = []
    for j in range(len(random_output)):
        polygon_types += [j] * random_output[j]
    np.random.shuffle(polygon_types)
    return polygon_types


def _incompatible_polygons(polygons):
    def overlap(pol1, pol2):
        return any(pixel in pol2 for pixel in pol1)

    def touch(pol1, pol2):
        return any(abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1
                   for p1, p2 in product(pol1, pol2))

    def incompatible(pol1, pol2):
        return overlap(pol1, pol2) or touch(pol1, pol2)

    return any(incompatible(c[0], c[1]) for c in combinations(polygons, 2))


def _polygon_generator_type0(img_width, img_height, area):
    # generate triangle

    top_x = int(np.random.uniform(5, img_width - 5))
    top_y = int(np.random.uniform(1, img_height - 5))
    last_layer = [(top_x, top_y)]
    pixels = last_layer
    for j in range(4):
        new_layer = [(x, y + 1) for (x, y) in last_layer]
        new_layer = [(last_layer[0][0] - 1, last_layer[0][1] + 1)] + new_layer
        new_layer.append((last_layer[-1][0] + 1, last_layer[-1][1] + 1))
        last_layer = new_layer
        pixels += last_layer
    assert len(pixels) == area
    return pixels


def _polygon_generator_type1(img_width, img_height, area):
    # generate square

    top_x = int(np.random.uniform(1, img_width - 5))
    top_y = int(np.random.uniform(1, img_height - 5))
    pixels = [(top_x + i, top_y + j) for i, j in product(range(5), range(5))]
    assert len(pixels) == area
    return pixels


def _polygon_generator_type2(img_width, img_height, area):
    # generate rhombus

    top_x = int(np.random.uniform(4, img_width - 4))
    top_y = int(np.random.uniform(1, img_height - 7))
    last_layer = [(top_x, top_y)]
    pixels = last_layer
    symmetric = [(top_x, top_y + 6)]
    for j in range(3):
        new_layer = [(x, y + 1) for (x, y) in last_layer]
        new_layer = [(last_layer[0][0] - 1, last_layer[0][1] + 1)] + new_layer
        new_layer.append((last_layer[-1][0] + 1, last_layer[-1][1] + 1))
        if j < 2:
            symmetric += [(x, y + 4 - 2 * j) for (x, y) in new_layer]
        last_layer = new_layer
        pixels += last_layer
    pixels += symmetric
    assert len(pixels) == area
    return pixels


def _touch(polygon1, polygon2):
    for p1 in polygon1:
        for p2 in polygon2:
            if abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1:
                return True
    return False


def main(experiment_path, parity_check):
    experiment = load_experiment_config(experiment_path)
    logger = experiment.logger
    msg = "Going to generate {} images in {} folder..."
    logger.info(msg.format(experiment.dataset_size, experiment.dataset_folder))

    random.seed(experiment.dataset_seed)
    np.random.seed(experiment.dataset_seed)

    os.makedirs(experiment.dataset_folder, exist_ok=True)

    fns = get_module_functions(__name__)
    generated_images = set()

    for j in range(experiment.dataset_size):
        while True:
            shapes = _get_random_shapes(experiment.polygons_number,
                                        experiment.polygons_prob)
            if len(shapes) == len(np.unique(shapes)):  # only different shapes
                break

        image = _generate_image(experiment.img_width, experiment.img_height,
                                experiment.area, shapes, fns, generated_images,
                                parity_check)

        # output format: img_SAMPLENUMBER_[POLYGONTYPES].png
        file_name = "img_{}_{}.png"
        image_filename = file_name.format(j, str(shapes).replace(" ", ","))
        image.save(experiment.dataset_folder + image_filename)
        if (j + 1) % 100 == 0:
            logger.info("{} images generated so far...".format(j + 1))

    logger.info("Compressing dataset in zip archive...")
    make_archive(experiment.datasets_folder + experiment.dataset_name, 'zip',
                 root_dir=experiment.dataset_folder)
    logger.info("Done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the experiment .json")
    parser.add_argument("-p", "--parity", type=bool, default=False,
                        help="Flag to add parity checker pixels")

    args = parser.parse_args()
    main(args.input, args.parity)
