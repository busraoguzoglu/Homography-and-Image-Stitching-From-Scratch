import PIL
from PIL import Image, ImageChops


def stitch_manual_to_left(img1, img2, offset_x, offset_y):

    #img2 = trim_left(img2, offset_x)
    #img2.show()

    images = [img1, img2]
    widths, heights = zip(*(i.size for i in images))

    # To find the offset we need to find where the corners of second img go in first img
    # x_bias, y_bias can be used.

    # biasses are wrong. x -> y and y -> x

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width-offset_x, max_height-offset_y))  # Total width - offset

    offset_x_sum = 0
    offset_y_sum = 0
    new_im.paste(img1, (offset_x_sum, offset_y_sum))

    offset_x_sum += offset_x
    offset_y_sum += offset_y
    new_im.paste(img2, (offset_x_sum, offset_y_sum))

    return new_im


def stitch_manual_to_right(img1, img2, offset_x, offset_y):

    # second image is trimmed according to offset
    #img2 = trim_left(img2, offset_x)
    # first image also trimmed.
    #img1 = trim_right(img1, offset_x)

    img1.show()
    img2.show()

    images = [img1, img2]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (img2.size[0]+offset_x, max_height - offset_y))  # Total width - offset

    offset_x_sum = 0
    offset_y_sum = 0

    # First, second image will be put to rightmost location.
    new_im.paste(img2, (offset_x_sum, offset_y_sum))
    # Okay until here.

    offset_x_sum += 0
    offset_y_sum += offset_y

    new_im.paste(img1, (offset_x_sum, offset_y_sum))

    return new_im


def stitch_manual_to_right_and_left(img1, img2, img3, offset_xleft, offset_yleft, offset_xright, offset_yright):
    # second image is trimmed according to offset
    # img2 = trim_left(img2, offset_x)
    # first image also trimmed.
    # img1 = trim_right(img1, offset_x)

    images = [img1, img2]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))  # Total width - offset

    offset_x_sum = total_width-img2.size[0]
    offset_y_sum = offset_yright+offset_xright

    # First, second image will be put to rightmost location.
    new_im.paste(img2, (offset_x_sum, offset_y_sum))
    # Okay until here.

    offset_x_sum += 0
    offset_y_sum += offset_yright

    middle_x_location = offset_x_sum
    middle_y_location = offset_y_sum

    offset_x_sum -= offset_xleft
    offset_y_sum -= offset_yleft

    new_im.paste(img3, (offset_x_sum, offset_y_sum))
    new_im.paste(img1, (middle_x_location, middle_y_location))

    return new_im

def trim_left(im, offset):

    w = (int)(im.size[0])
    h = (int)(im.size[1])

    img_right_area = (offset, 0, w, h)
    img_right = im.crop(img_right_area)

    return img_right


def trim_right(im, offset):

    w = (int)(im.size[0])
    h = (int)(im.size[1])

    img_left_area = (0, 0, offset, h)
    img_left = im.crop(img_left_area)
    img_left.show()
    return img_left


# Main function definition
def main():

    # Warped images and offsets are provided
    # for simple testing.

    img1 = Image.open('./images/paris_with_warps/paris_b.jpg')
    img2 = Image.open('./images/paris_with_warps/pariscwarp.jpg')
    img3 = Image.open('./images/paris_with_warps/parisawarp.jpg')

    # Offsets got from points.py
    result_image = stitch_manual_to_right_and_left(img1, img2, img3, 903, 229, 236, 40)
    result_image.show()
    result_image.save('./images/paris_with_warps/parisfullstitch.jpg')


# Call main function
if __name__ == '__main__':
    main()