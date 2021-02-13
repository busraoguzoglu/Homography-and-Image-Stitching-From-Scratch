from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import points

def main():

    # For fast testing:
    # Points for warping paris_a and paris_c
    # Were specified in points.py
    # Initially, paris_c will be warped when this program runs.
    # To warp paris_a, simply change points in homography to:
    # points.pts_src_hand_select1, points.pts_dst_hand_select1

    im1 = cv2.imread('./images/paris_with_warps/paris_c.jpg')
    im2 = cv2.imread('./images/paris_with_warps/paris_b.jpg')

    # For Gaussian Noise Test
    #im1 = add_noise(im1)
    #print('noise added')
    #cv2.imwrite('./results/noise_adding_warp_test/noise_picture.jpg', im1)

    # Selecting points by hand
    # To test this feature, change the points in homography function
    #pts_src1, pts_dst1 = select_points(im1, im2, 4)

    # Finding homography
    h1 = computeH(points.pts_src_hand_select2, points.pts_dst_hand_select2)

    # Warping Image
    result3, x_bias, y_bias, warped_corners = warp(im1, h1)

    # Plotting First image, second image, result
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(result3)
    plt.show()

    # Provide filepath for warped picture to save
    # Save filepath was specified to make stitch test easier.
    # However, warped pictures will be provided if this will not run.
    #save_filepath = './images/paris/pariscwarp.jpg'
    #cv2.imwrite(save_filepath, result3)


# This function will compute the homography matrix between two images
# using selected point pairs.
def computeH(points_im1, points_im2):

    num_of_correspondence = int(points_im1.size)/2
    compute_range = num_of_correspondence.__int__()-1

    # For each correspondence form matrix:

    x1, y1 = points_im1[0]
    x2, y2 = points_im2[0]

    a1 = [x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2]
    a2 = [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2]

    # Put them in a 2x9 array
    matrix = np.asmatrix([a1, a2])

    for i in range(compute_range):

        x1, y1 = points_im1[i+1]
        x2, y2 = points_im2[i+1]

        a1 = [x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2, -x2]
        a2 = [0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2, -y2]

        matrix = np.vstack([matrix, a1])
        matrix = np.vstack([matrix, a2])

    # can use svd func.

    u, s, v = np.linalg.svd(matrix)
    h = np.reshape(v[8], (3, 3))
    h = (1 / h.item(8)) * h

    return h


# This function will do warp operation.
# Takes up to five minutes to give the result.
# Printing offsets before warp starts,
# If it is too big, stop the process
# Either points are very wrong
# Or image is too big, it will take long time.
def warp(image, homography):

    # for all pixels in the image:
    # we need to change coordinates.
    # get pixels in all coordinates one by one, put them in corresponding positions:
    # do this by backward transform:
    # go from target -> source by inverse homo

    h = image.shape[0]
    w = image.shape[1]
    homo_inv = np.linalg.inv(homography)

    # Finding warped image size from source image corners + homo
    corners = np.array([[0, 0], [h - 1, 0], [0, w - 1], [h - 1, w - 1]])
    p = np.vstack([corners.transpose(), [1, 1, 1, 1]])

    #   1   2   3   4
    #[  0 251   0 251]
    #[  0   0 256 256]
    #[  1   1   1   1]

    q = (homography.dot(p))
    q = np.delete((q / q[2]), 2, 0)
    warped_corners = np.round(q).astype(np.int32)

    return_dimensions = warped_corners.reshape(-1, 1, 2)
    print('warped_Corners: ', return_dimensions)

    max_y, max_x = np.max(warped_corners, axis=1)
    min_y, min_x = np.min(warped_corners, axis=1)

    x_bias = -min(0, min_x)
    y_bias = -min(0, min_y)

    print('min x:', min_x, 'max_x:', max_x, 'min y:', min_y, 'max y:', max_y)

    # Return offset numbers

    x_offset = x_bias
    y_offset = y_bias

    if x_bias == 0:
        x_offset = -min_x

    if y_bias == 0:
        y_offset = -min_y

    print('x_bias:', x_offset)
    print('y_bias:', y_offset)

    warp_w = np.arange(min(0, min_x), max(max_x + 1, h))   # width: coordinates of warped image
    warp_h = np.arange(min(0, min_y), max(max_y + 1, w))   # height: coordinates of warped image

    # creation of warped image base
    warped_image = np.zeros((len(warp_w), len(warp_h), image.shape[2]), dtype=np.uint8)

    # Process every pixel backwards (from warped to original)

    print('warp starts')

    for i in warp_w:
        for j in warp_h:

            p = np.array([i, j]).reshape(2, -1)
            p = np.vstack([p, [1]])

            # rearrange (did not understood, does not work if this step is missed.)
            temp = p[1][0]
            p[1][0] = p[0][0]
            p[0][0] = temp

            q = homo_inv.dot(p)
            q = np.delete((q / q[2]), 2, 0)
            q = np.round(q).astype(np.int32)

            # Corresponding x,y in first image
            x_first, y_first = (q[::-1, :])

            # Check if this points lie in source image
            if x_first < h and x_first > 0 and y_first < w and y_first > 0:
                current_color = image[x_first, y_first]
            else:
                current_color = 0

            # Put pixel in warped location
            x_new = i+int(x_bias)
            y_new = j+int(y_bias)

            warped_image[x_new][y_new] = current_color

    return warped_image, x_offset, y_offset, return_dimensions


# There is a known bug in this method.
# Sometimes while selecting points, it gives memory error.
# This may be caused by ginput or other array related bug.
# Try again to solve it, it works.
def select_points(im1, im2, number_of_corresponding_points):

    plt.subplot(1, 2, 1)
    plt.imshow(im1)

    plt.subplot(1, 2, 2)
    plt.imshow(im2)

    correspondence_image_1_2 = plt.ginput(2 * number_of_corresponding_points, show_clicks=True)
    print("clicked", correspondence_image_1_2)

    points_im1_to2 = np.empty(shape=(number_of_corresponding_points, 2))  # points_im1_to2[0] correspond to points_im2_to1[0]
    points_im2_to1 = np.empty(shape=(number_of_corresponding_points, 2))

    for i in range(number_of_corresponding_points):
        points_im1_to2[i] = (correspondence_image_1_2[2 * i])
        points_im2_to1[i] = (correspondence_image_1_2[2 * i + 1])

    print('points_im1_to2', points_im1_to2)
    print('points_im2_to1', points_im2_to1)

    return points_im1_to2, points_im2_to1


# Utility function to add Gaussian Noise
def add_noise(im):

    noise_im = np.zeros(im.shape, np.uint8)
    mean = (20,20,20)
    sigma = (20,20,20)
    cv2.randn(noise_im, mean, sigma)
    noise_image = cv2.add(im, noise_im)
    plt.imshow(noise_image)

    save_filepath = './results/paris/noise_img.jpg'
    cv2.imwrite(save_filepath, noise_image, )

    print('im saved')

    return noise_image


if __name__ == '__main__':
    main()