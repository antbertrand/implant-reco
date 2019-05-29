




import os
import cv2

from gen_data import generate_chip




def create_chips(dest_folder="."):

    nb_chips = 3
    data = []

    for k in range(nb_chips):

          chip, coords, caracs, letters_size = generate_chip()
          path_name = os.path.join(dest_folder, "GENERATED_CHIP-{}.jpg".format(k))
          cv2.imwrite(path_name, chip)
          data.append({"path": os.path.abspath(path_name), "coords": coords, "caracs": caracs, "carac_size": letters_size})
          write_csv(data)





def write_csv(data, file_name = "ann_all.csv"):

    if not os.path.isfile(file_name):
        with open(file_name, 'w'):
            pass

    with open(file_name, 'a') as f:
        for datum in data:
            for i in range(len(datum['coords'])):
                coord = "{},{},{},{}".format(
                    datum['coords'][i][0],
                    datum['coords'][i][1],
                    datum['coords'][i][0]+datum['carac_size'][0],
                    datum['coords'][i][1]+datum['carac_size'][1],
                )
                f.write(",".join([datum['path'], coord, datum["caracs"][i]]))
                f.write("\n")

create_chips()
