

import os
import numpy as np
import cv2

from gen_data import generate_chip




def create_chips(dest_folder=".", n=1000, side_length=224):
    """ Create n artificial chips
    """

    for k in range(n):
        data = []
        # Generate a chip and set a path name
        chip, boxes, caracs = generate_chip()
        path_name = os.path.join(dest_folder, "GENERATED_CHIP-{}.jpg".format(k))

        # Resize the chip and adapt the coordinates
        scale = chip.shape[0] / side_length
        chip = cv2.cvtColor(chip, cv2.COLOR_RGB2GRAY)
        chip = cv2.equalizeHist(chip)
        chip = cv2.resize(chip, (side_length, side_length))

        boxes = np.array(boxes)
        boxes = boxes.astype(float)
        boxes /= scale


        # Save the image
        if k<n*0.8:
            path_name = os.path.join(dest_folder, "train", "GENERATED_CHIP-{}.jpg".format(k))
            file_name = "/storage/eurosilicone/ds_step4/gen_dataset/ann_train.csv"

        else:
            path_name = os.path.join(dest_folder, "val", "GENERATED_CHIP-{}.jpg".format(k))
            file_name = "/storage/eurosilicone/ds_step4/gen_dataset/ann_val.csv"

        cv2.imwrite(path_name, chip)
        # Set all the data we need for annotation
        data.append({"path": os.path.abspath(path_name), "coords": boxes.astype(int), "caracs": caracs})

        # Write the annotation of this chip in the csv file
        write_csv(data, file_name)






def write_csv(data, file_name = "ann_all.csv"):
    """ Write a csv file for annotations of chips
    """
    # If the file doesn't exist, create it
    if not os.path.isfile(file_name):
        with open(file_name, 'w'):
            pass

    # Then append, the annotations of the chip
    with open(file_name, 'a') as f:
        for datum in data:
            for i in range(len(datum['coords'])):
                coord = "{},{},{},{}".format(
                    datum['coords'][i][0],
                    datum['coords'][i][1],
                    datum['coords'][i][2],
                    datum['coords'][i][3],
                )
                # Write the path of the image, the coordinates and the type of each letter
                f.write(",".join([datum['path'], coord, datum["caracs"][i]]))
                f.write("\n")

with open("/storage/eurosilicone/ds_step4/gen_dataset/ann_train.csv", 'w'):
    pass
with open("/storage/eurosilicone/ds_step4/gen_dataset/ann_val.csv", 'w'):
    pass
create_chips(dest_folder='/storage/eurosilicone/ds_step4/gen_dataset/')
