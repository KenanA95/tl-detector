import yaml
from train import load_descriptor
from detectors import *
from helpers import read_directory_images, cutoff_lower
from src.heatmap import HeatMap
from sklearn.externals import joblib
from ast import literal_eval


def load_detector(settings):
    return {
        'spotlight': SpotlightDetector.from_config_file(settings['spotlight_detector']),
        'color': ColorDetector.from_config_file(settings['color_detector'])
    }.get(settings['run']['detector'], 'spotlight')  # Default to spotlight for invalid input


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        settings = yaml.load(stream)

    detector = load_detector(settings)
    descriptor = load_descriptor(settings)
    classifier = joblib.load(settings['run']['classifier_location'])

    # Grab the shape of the first image in the directory to determine the size of the heatmap
    heatmap_shape = list(read_directory_images(settings['run']['image_directory'], extension='', n=1))[0].shape[:2]
    heatmap = HeatMap(heatmap_shape, int(settings['run']['heatmap_memory']), int(settings['run']['heatmap_threshold']))

    win_size = literal_eval(settings['run']['window_size'])
    x_offset = win_size[0] / 2
    y_offset = win_size[1] / 2

    images = read_directory_images(settings['run']['image_directory'], extension='')

    for image in images:
        top_half = cutoff_lower(image, 0.45)
        lights = classifier.run_detector(detector, top_half, win_size)

        bounding_boxes = []

        for (x, y) in lights:
            box = {'x_min': (x-x_offset), 'x_max': (x+x_offset), 'y_min': (y-y_offset), 'y_max': (y+y_offset)}
            bounding_boxes.append(box)

        heatmap.update(bounding_boxes)
        heatmap.draw(image)
        cv2.imshow('labelled image', image)
        cv2.waitKey(10)
