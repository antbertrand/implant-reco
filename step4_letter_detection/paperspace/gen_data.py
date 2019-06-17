#!/usr/bin/env python
# coding: utf-8


import os
import glob

import numpy as np
import random

import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter

from deterioration import add_parallel_light, noisy


BG_PATH = './backgrounds/'
CARAC_PATH = './caracters/'


img = cv2.imread(BG_PATH + '4.png')
overlay_t = cv2.imread(CARAC_PATH + 'letter_4.png', -
                       1)  # -1 loads with transparency


#print(overlay_t.shape)


def overlay_caracter(background_img, caracter_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      caracter_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        caracter_t = cv2.resize(caracter_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    # print(caracter_t.shape)
    b, g, r, a = cv2.split(caracter_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                              mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


def generate_letter(letter, angle=0):
    """ Generate a textured letter
    """
    img = Image.new("RGB", (140, 180), "black")
    # get a font
    # get a drawing context
    d = ImageDraw.Draw(img)

    # Select the right font for the right letters
    if letter == "3":
        fnt = ImageFont.truetype('./fonts/AVHersheySimplexMedium.otf', 218)
        # draw text, full opacity
        d.text((3, 10), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((7, 7), np.uint8)
    elif letter == "1":
        fnt = ImageFont.truetype('./fonts/Calibri.ttf', 200)
        # draw text, full opacity
        d.text((8, -30), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((10, 10), np.uint8)
    elif letter == "W":
        fnt = ImageFont.truetype('./fonts/Goodlight-Light.otf', 110)
        # draw text, full opacity
        d.text((2, 40), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((3, 3), np.uint8)
    elif letter == "I":
        fnt = ImageFont.truetype('./fonts/Tahoma.ttf', 180)
        # draw text, full opacity
        d.text((20, -15), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)
    elif letter == "M":
        fnt = ImageFont.truetype('./fonts/Tahoma.ttf', 180)
        # draw text, full opacity
        d.text((0, -20), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)
    elif letter in "02456789":
        fnt = ImageFont.truetype('./fonts/Arial.ttf', 180)
        # draw text, full opacity
        d.text((20, 0), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)
    elif letter == "/":
        fnt = ImageFont.truetype('./fonts/Arial.ttf', 180)
        # draw text, full opacity
        d.text((30, 0), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)
    else:
        fnt = ImageFont.truetype('./fonts/Arial.ttf', 180)
        # draw text, full opacity
        d.text((2, 0), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)

    # Rotate the image if needed
    if angle > 1:
        img = img.transpose(angle)

    # Blur and erode the letter so they are thinner
    img = np.array(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.erode(img, kernel, iterations=1)
    img = Image.fromarray(img)

    # Make an emboss effect
    img = img.filter(ImageFilter.EMBOSS)

    # Rerotate the image if needed
    if angle > 1:
        img = img.transpose(6 - angle)

    # Resize the image so the letter is narrow
    img = img.resize((100, 180))

    # Set the gray pixels as transparent
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    alpha_value = random.randint(120, 150)
    for item in datas:
        if item[0] == 128 and item[1] == 128 and item[2] == 128:
            newData.append((255, 255, 255, 0))
        else:
            # Chooses a random value for the alpha channel
            newData.append(
                (item[0], item[1], item[2], alpha_value))

    img.putdata(newData)

    # Resize the image so the letter has the same size than on the chip
    img = img.resize((150, 266))

    # if letter == '/':
    #     img.save('eurosilicone_letters/letter_{}.png'.format('slash'), "PNG")
    # else:
    #     img.save('eurosilicone_letters/letter_{}.png'.format(letter), "PNG")

    # Crop borders
    img = img.crop((5, 20, 147, 260))
    #bbox = img.getbbox()
    #img = img.crop(bbox)

    # img = np.array(img)
    #
    # img = add_gaussian_noise(img, 25)
    #
    # cv2.imwrite("step4.png", img)
    #
    # img = img[20:img.shape[0]-1, 1:img.shape[1]-1]
    #
    # img = img.astype(int)

    # print(type(img))
    #img.save('.letter.png')
    return img, img.size


# In[175]:


def print_text(dispo, im):

    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
    #numbers = '0123456789'
    [width, height] = im.size
    ratio_line = [0.42, 0.60, 0.75]
    boxes = []
    caracs = []
    im = im.convert("RGB")
    im2 = np.array(im)
    #print("IMMM", im2.shape)

    for index, line in enumerate(dispo):

        len_line = len(line[0])

        width_carac = 150
        start_offset = int(width / 2 - (len_line * width_carac) / 2)

        y = int(ratio_line[index] * height)

        for index2, carac_type in enumerate(line[0]):

            if carac_type == 'l':
                carac = random.choice(letters)
            elif carac_type == 'n':
                carac = random.choice(numbers)
            elif carac_type == 'c':
                carac = 'C'
            elif carac_type == '/':
                carac = '/'
            else:
                print('Hmm, not valid caracter in disposition')

            x = start_offset + width_carac * index2

            caracter_t, letters_size = generate_letter(carac)
            caracter_t.save('./carac.png')

            #print(im.mode)
            im.paste(caracter_t, (x, y), caracter_t)

            caracter_t_cv = np.array(caracter_t)
            cv2.imwrite('./carac_cv.png', caracter_t_cv)


            #print(caracter_t_cv.shape)
            channels = cv2.split(caracter_t_cv)
            (x0 , y0 ,w ,h) = cv2.boundingRect(channels[3])
            #x_correc = x + int((letters_size[0] - w)/2) - x0
            #y_correc = y + int((letters_size[1] - h)/2) - y0
            a = x + x0 + int(width_carac-letters_size[0])- 10
            b = y0 + y
            #box = [x_correc, y_correc, x_correc+w, y_correc+h]
            box = [ a, b, a+w, b+h]
            #print(box)
            boxes.append(box)
            caracs.append(carac)

            #im = overlay_caracter(im, caracter_t, x, y, overlay_size=None)

    return im, boxes, caracs


def generate_chip():

    BG_PATH = './backgrounds/'
    bgs = glob.glob('{}*.png'.format(BG_PATH))
    # Different dispositions on the chip for the different lines :  'n' : number
    #                                                               'c' : letter c
    #                                                               'l' : letter
    dispositions = [['llnnn', ],
                    ['nn/nnnll'],  # 'nnl/nnncc','nnl/nnncc','lln/nnncc',],
                    ['lnnnn', ]]

    dispositions2 = [['lllll', ],
                     ['llllllll'],  # 'nnl/nnncc','nnl/nnncc','lln/nnncc',],
                     ['lllll', ]]
    n = random.randint(0, 3)

    # Randomly choose dispo
    dispo = dispositions2
    #dispo[1] = [dispositions[1][n]]

    # Randomly choose background
    #bg = cv2.imread(BG_PATH + bgs[random.randint(0,6)])
    bg = Image.open(random.choice(bgs))
    bg = bg.resize((1580, 1580))
    chip, boxes, caracs = print_text(dispo, bg)

    chip_cv = np.array(chip)

    #To print boxes on image
    for box in boxes:
        cv2.rectangle(chip_cv,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)

    # Add deteriorations
    chip_final, _ = add_parallel_light(chip)
    chip_final = noisy('gauss', chip_final)
    chip_final = noisy('s&p', chip_final)

    #cv2.imwrite('./CHIP.png', chip_final)

    return chip_final, boxes, caracs


generate_chip()
