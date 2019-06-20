import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import cv2
import matplotlib.pyplot as plt
from random import choice

def add_gaussian_noise(image_in, noise_sigma):
    """ Add random noise the image
    """
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    return noisy_image

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
        d.text((3, 20), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((7, 7), np.uint8)
    elif letter == "1":
        fnt = ImageFont.truetype('./fonts/Calibri.ttf', 226)
        # draw text, full opacity
        d.text((8, -40), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((10, 10), np.uint8)
    elif letter == "W":
        fnt = ImageFont.truetype('./fonts/Goodlight-Light.otf', 110)
        # draw text, full opacity
        d.text((2, 40), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((3, 3), np.uint8)
    elif letter == "I":
        fnt = ImageFont.truetype('./fonts/Tahoma.ttf', 180)
        # draw text, full opacity
        d.text((20, -10), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)
    elif letter == "M":
        fnt = ImageFont.truetype('./fonts/Tahoma.ttf', 180)
        # draw text, full opacity
        d.text((0, -20), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)
    else:
        fnt = ImageFont.truetype('./fonts/Arial.ttf', 188)
        # draw text, full opacity
        d.text((2, 0), letter, font=fnt, fill=(255, 255, 255, 255))
        kernel = np.ones((9, 9), np.uint8)

    # Rotate the image if needed
    if angle > 1:
        img = img.transpose(angle)

    # Blur and erode the letter so they are thinner
    img = np.array(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.erode(img, kernel, iterations = 1)
    img = Image.fromarray(img)

    # Make an emboss effect
    img = img.filter(ImageFilter.EMBOSS)

    # Rerotate the image if needed
    if angle > 1:
        img = img.transpose(6-angle)

    # Resize the image so the letter is narrow
    img = img.resize((100, 180))

    # Set the gray pixels are transparent
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 128 and item[1] == 128 and item[2] == 128:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)

    # Resize the image so the letter has the same size than on the chip
    img = img.resize((159, 282))

    # if letter == '/':
    #     img.save('eurosilicone_letters/letter_{}.png'.format('slash'), "PNG")
    # else:
    #     img.save('eurosilicone_letters/letter_{}.png'.format(letter), "PNG")

    # Crop borders
    img = img.crop((5, 20, 155, 275))

    # img = np.array(img)
    #
    # img = add_gaussian_noise(img, 25)
    #
    # cv2.imwrite("step4.png", img)
    #
    # img = img[20:img.shape[0]-1, 1:img.shape[1]-1]
    #
    # img = img.astype(int)

    return img

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890/":
#for letter in "AIMW13":

    # Set an angle for rotation 1=no rotation, 4=270° rotation
    angle = choice([1, 2, 3, 4])

    # Generate the letter
    img = generate_letter(letter, angle)

    # Save the image (deal with "/" in path)
    if letter == '/':
        img.save('eurosilicone_letters/letter_{}.png'.format('slash'), "PNG")
    else:
        img.save('eurosilicone_letters/letter_{}.png'.format(letter), "PNG")

    # plt.imshow(img, cmap='gray')
    # plt.show()
