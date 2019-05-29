




import os
import numpy as np
import cv2

from gen_data import generate_chip




def create_chips(dest_folder=".", n=1024, side_length=224):
    """ Create n artificial chips
    """
    data = []
    for k in range(n):
        # Generate a chip and set a path name
        chip, boxes, caracs = generate_chip()
        path_name = os.path.join(dest_folder, "GENERATED_CHIP-{}.jpg".format(k))

        # Resize the chip and adapt the coordinates
        scale = chip.shape[0] / side_length
        chip = cv2.resize(chip, (side_length, side_length))
        boxes = np.array(boxes)
        boxes *= scale

        # Save the image
        cv2.imwrite(path_name, chip)

        # Set all the data we need for annotation
        data.append({"path": os.path.abspath(path_name), "coords": boxes.astype(int), "caracs": caracs})

        # Write the annotation of this chip in the csv file
        write_csv(data)





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

create_chips(dest_folder=".")
