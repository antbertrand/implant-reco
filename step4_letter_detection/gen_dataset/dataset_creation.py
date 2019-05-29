





from gen_data import generate_chip




def create_chips():

    nb_chips = 1024

    for k in range(nb_chips):

          chip, coords, caracs, letters_size = generate_chip()
          write_csv ( coords, letters_size)





def write_csv():


    resize_shape = (300, 450)

    with open('ann_all.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)


    shape = (x['size']['height'],x['size']['width'])
    path_to_image = IMAGE_PATH + true_filename

    class_name = x['objects'][0]['classTitle']
    
    spamwriter.writerow([path_to_image,x1,y1,x2,y2,class_name])