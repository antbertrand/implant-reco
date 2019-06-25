"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os
import pandas as pd

import cv2
import progressbar
assert(callable(progressbar.progressbar)
       ), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes(
    )) if generator.has_label(i)] for j in range(generator.size())]
    dataset_detections = []
    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        # ab : ordered from biggest to lowest
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(
            image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)



        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(
                i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores,
                            image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections ( ordered by label)
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]


        # copy detections to dataset_detection ( ordered by image )
        dataset_detections.append([image_boxes, image_scores, image_labels])

    return all_detections, dataset_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(
    generator.num_classes())] for j in range(generator.size())]

    dataset_annotations = []
    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations ( order by label)
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels']
                                                              == label, :].copy()


        # copy detections to dataset_detection ( ordered by image )
        dataset_annotations.append([annotations['bboxes'], annotations['labels']] )

    return all_annotations, dataset_annotations


def get_code(boxes, scores, labels, score_threshold = 0.2):
    """docstrung"""
    print("Getcode")
    labels_to_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                       6: '6', 7: '7', 8: '8', 9: '9', 10: '/', 11: 'A', 12: 'B', 13: 'C',
                       14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K',
                       22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R',
                       29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'}

    # Variable initialization
    compteur = 0
    current_grp = 0

    # Stores the anchor value for each line
    # anchor = [anchor_line1, anchor_line1, anchor_line1]
    anchor = [0, 0, 0, 0, 0]

    # Sorting the boxes by the y value of the top left coordinate
    infos = zip(boxes, scores, labels)
    infos_sorted = sorted(infos, key=lambda x: x[0][1])

    # infos_lines = [infos_line1, infos_line2, infos_line3]
    infos_lines = [[], [], [], [], []]

    # Passing through all the boxes
    for box, score, label in infos_sorted:

        y = box[1]

        # Filtering some boxes with a too low score and the -1 values that come from padding.
        #score_threshold = 0.2
        if label != -1 and score > score_threshold:

            # To initate anchor
            if compteur == 0:
                anchor[current_grp] = y

            # Goes into that condition when another group starts
            if abs(y - anchor[current_grp]) > 50:
                current_grp += 1
                if current_grp > 6:
                    print("The caracters have been grouped in more than 3 lines")
                    break
                anchor[current_grp] = y

             # Goes into that condition when still in same group
            if abs(y - anchor[current_grp]) < 50:
                infos_lines[current_grp].append([box, score, label])
                anchor[current_grp] = y

            compteur += 1

    # Keeping only the 3 lines with the most caracters
    size_grp = [len(i) for i in infos_lines]
    sorted_size_index = sorted(range(len(size_grp)), reverse = True, key=lambda k: size_grp[k])
    index_final = sorted_size_index[0:3]
    index_final = sorted(index_final)

    infos_lines_final = [infos_lines[i] for i in index_final]

    # Sorting each lines on the x coordinates.
    # (aren't we reading from left to right ?)
    infos_lines_final[0] = sorted(infos_lines_final[0], key=lambda x: x[0][0])
    infos_lines_final[1] = sorted(infos_lines_final[1], key=lambda x: x[0][0])
    infos_lines_final[2] = sorted(infos_lines_final[2], key=lambda x: x[0][0])

    # Printing text
    lines = ['', '', '']
    # Line 1
    for infos_carac in infos_lines_final[0]:
        lines[0] += labels_to_names[infos_carac[2]]
    # Line 2
    for infos_carac in infos_lines_final[1]:
        lines[1] += labels_to_names[infos_carac[2]]
    # Line 3
    for infos_carac in infos_lines_final[2]:
        lines[2] += labels_to_names[infos_carac[2]]

    return lines

def read_csv_code():
    csv_path = '/home/numericube/Documents/current_projects/gcaesthetics-implantbox/dataset/ds_step4_caracter_detector/codes_test.csv'
    data_code = pd.read_csv(csv_path)

    return data_code

def get_true_code(im_path):
    print("image = ", os.path.basename(im_path))
    data_code = read_csv_code()
    chip_data = data_code.loc[data_code['image_name'] == os.path.basename(im_path)]
    code = [ chip_data.iloc[0,1], chip_data.iloc[0,2] ,chip_data.iloc[0,3]]
    quality = chip_data.iloc[0,4]
    read = chip_data.iloc[0,5]
    return code, quality, read

def compare_codes(generator, dataset_detections, dataset_annotations, score_threshold):
    """docstrung"""

    nb_correct_codes = 0
    nb_both = 0
    nb_noone = 0
    nb_human_notai = 0
    nb_ai_nothuman = 0

    if len(dataset_detections) != len(dataset_annotations):
        print("problem unequal size of dataset_ann and dataset_detec", len(dataset_detections) , len(dataset_annotations))


    for i in progressbar.progressbar(range(len(dataset_detections)), prefix='Comparing codes: '):
        print("NUMERO {}".format(i))
        image_detections = dataset_detections[i]
        image_annotations = dataset_annotations[i]

        # Get the detected code
        detected_code = get_code(image_detections[0], image_detections[1], image_detections[2], score_threshold=score_threshold)


        # Get the true codes
        # ( Score of 1 because true code)
        #true_code = get_code(image_annotations[0], [1 for i in range(len(image_annotations[0]))], image_annotations[1])
        #read_bool = 0
        true_code, im_quality, read_bool = get_true_code(generator.image_names[i])

        print("Detected =", detected_code)
        print("True     =", true_code)
        if detected_code == true_code:

            print('Youhou ! correct code', detected_code)
            nb_correct_codes += 1

            if read_bool == 1:
                nb_both += 1

            else:
                nb_ai_nothuman += 1

        else:
            #Let's try to find out what went wrong more precisely
            if read_bool == 1:
                nb_human_notai += 1

            else:
                nb_noone += 1

            print("notgood")
        print("==========================================")
    # Print good score accuracy
    print("There were {} correctly detected codes, acc : {}\n".format(nb_correct_codes, nb_correct_codes/len(dataset_annotations)))
    print("Out of the {} readable codes (seen by human eye), {} were read by the ai => {}% \n".format(nb_both + nb_human_notai, nb_ai_nothuman + nb_both, 100*(nb_ai_nothuman + nb_both)/(nb_human_notai+ nb_both)))
    print("There were {} correctly detected codes by ai and by a human eye\n".format(nb_both))
    print("There were {} correctly detected codes by ai but not by a human eye\n".format(nb_ai_nothuman))
    print("There were {} correctly detected codes by a human eye but not by the ai\n".format(nb_human_notai))
    print("There were {} falsely detected codes by a human eye and by the ai\n".format(nb_noone))
    return None




def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, dataset_detections = _get_detections(
        generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations, dataset_annotations = _get_annotations(generator)
    average_precisions = {}


    # Compare codes true and obtained codes
    compare_codes(generator, dataset_detections, dataset_annotations, score_threshold)



    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # process detections and annotations

    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(
                    np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / \
            np.maximum(true_positives + false_positives,
                       np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions
