

import os
import cv2

from gen_data import generate_chip


def create_chips(dest_folder=".", side_length=224):

    nb_chips = 10
    data = []

    for k in range(nb_chips):
        chip, boxes, caracs = generate_chip()
        path_name = os.path.join(
            dest_folder, "GENERATED_CHIP-{}.jpg".format(k))
        scale = chip.shape[0] / side_length
        chip = cv2.resize(chip, (side_length, side_length))
        boxes *= scale
        cv2.imwrite(path_name, chip)
        data.append({"path": os.path.abspath(path_name),
                     "coords": boxes, "caracs": caracs})
        write_csv(data)


def write_csv(data, file_name="ann_all.csv"):

    if not os.path.isfile(file_name):
        with open(file_name, 'w'):
            pass

    with open(file_name, 'a') as f:
        for datum in data:
            for i in range(len(datum['coords'])):
                coord = "{},{},{},{}".format(
                    datum['coords'][i][0],
                    datum['coords'][i][1],
                    datum['coords'][i][2],
                    datum['coords'][i][3],
                )
                f.write(",".join([datum['path'], coord, datum["caracs"][i]]))
                f.write("\n")


create_chips(dest_folder='./dataset_generated')
