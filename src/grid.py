"""Try to find a grid (or a chessboard) in the given image.
"""
from copy import deepcopy
import cv2
import math
import imutils
import numpy as np

debug_export_path = "/home/flyingdutchman/Bilder/"


class GridFinderException(Exception):
    """Exception thrown by GridFinder when somethings wrong"""
    pass


class Grid(object):
    """Given the result of a cv2.Canny, find a grid in the given image.
    The grid have no start and no end, only a "cell width" and an
    anchor (an intersection, so you can anlign the grid with the image).
    Exposes a list of columns (x, width) and a list of rows (y, height),
    as self.all_x and self.all_y.
    Exposes a all_cells() method, yielding every cells as tuples
    of (x, y, width, height).
    And a draw(self, image, color=(255, 0, 0), thickness=2) method,
    to draw the grid on a given image, usefull to check for correctness.
    """


def parse_args(args=None):
    """Return parsed arguments from command line"""
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Grid finder')
    parser.add_argument('file', help='Input image')
    parser.add_argument('--hc', help='Color of the human player.'
                                     'Default = red', default='red')
    parser.add_argument('--rc', help='Color of the robot player.'
                                     'Default = green', default='green')
    parser.add_argument('--bb', help='Buffer to add around detected rectangle to improve grid detection.'
                                     'Default = 1', default='1')
    parser.add_argument('--resize', help='Resizes the input image for better performance.'
                                         'Default = 500', default='500')
    parser.add_argument('--saturation', help='Modify the saturation of the input image.'
                                             'Default = 1', default='1')
    if args is None:
        return parser.parse_args()
    return parser.parse_args(args)


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # warp_image(image, tl, tr, br, bl)
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    converted_red_pixel_value = [0, 0]
    converted_green_pixel_value = [maxWidth, 0]
    converted_black_pixel_value = [0, maxHeight]
    converted_blue_pixel_value = [maxWidth, maxHeight]

    point_matrix = np.float32([tl, tr, br, bl])

    # Convert points
    converted_points = np.float32([converted_red_pixel_value, converted_green_pixel_value, converted_black_pixel_value,
                                   converted_blue_pixel_value])

    # perspective transform
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    img_Output = cv2.warpPerspective(image, perspective_transform, (maxWidth, maxHeight))
    return img_Output


def grid_as_json(img, grid, human_color, robot_color, saturation: float):
    """Export the given grid as a json file containing a list of cells as:
    {'x': ..., 'y': ..., 'color': ...}
    """
    import json
    columns = {}

    #img = cv2.imread('/home/flyingdutchman/IdeaProjects/Bildverarbeitung2/oben_warped.jpg')
    #img_flat = img.copy()

    if img.shape[1] > 500:
        img_flat = imutils.resize(img, 500)
    else:
        img_flat = img.copy()
    # img_flat = cv2.rotate(img_flat, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_flat = increase_brightness(img_flat, 40)
    img_flat = modify_saturation(img_flat, saturation)

    #cv2.imwrite(debug_export_path + 'premod.jpg', img)
    #cv2.imwrite(debug_export_path + 'postmod.jpg', img_flat)

    # height, width, number of channels in image
    height = img_flat.shape[0]
    width = img_flat.shape[1]
    cell_width = (width / 7)
    cell_height = (height / 6)

    for column in range(1, 8):
        rows = {}
        for row in range(1, 7):
            y = math.ceil((cell_width * column) - cell_width)
            x = math.ceil((cell_height * row) - cell_height)
            img_flat[x:x + 5, y:y + 5] = [0, 0, 0]

            # We only want a smaller area of the cell
            center_x = x + int(cell_height / 2)
            center_y = y + int(cell_width / 2)
            offset_x_left = center_x - int(cell_height / 3)
            offset_x_right = center_x + int(cell_height / 3)
            offset_y_top = center_y - int(cell_width / 3)
            offset_y_bottom = center_y + int(cell_width / 3)

            average = img_flat[offset_x_left:offset_x_right, offset_y_top:offset_y_bottom].mean(axis=0).mean(axis=0)
            # average = cv2.mean(img_flat[x:x + cell_width, y:y + cell_height])

            color = find_base_color(average)
            rows['Row' + (str(row))] = color_to_player(color, human_color, robot_color)
            #if color != [255, 255, 255]:
            img_flat[offset_x_left:offset_x_right, offset_y_top:offset_y_bottom] = color
            img_flat[center_x - 5:center_x + 5, center_y - 5:center_y + 5] = [0, 0, 0]

        columns['Column' + (str(column))] = deepcopy(rows)

    cv2.imwrite(debug_export_path + 'final.jpg', img_flat)
    return json.dumps(columns, indent=4)


def color_to_player(color, human_color, robot_color):
    if color == [255, 0, 0]:
        color_string = 'blue'
    elif color == [0, 255, 0]:
        color_string = 'green'
    elif color == [0, 0, 255]:
        color_string = 'red'
    else:
        color_string = 'white'

    if color_string == human_color:
        return 'h'
    elif color_string == robot_color:
        return 'r'
    else:
        return '0'


def find_base_color(mean_color):
    low_red = [0, 30, 20]
    high_red = [18, 255, 255]
    low_red_2 = [165, 35, 20]
    high_red_2 = [180, 255, 255]
    low_blue = [90, 50, 20]
    high_blue = [135, 255, 255]
    low_green = [35, 20, 20]
    high_green = [105, 255, 255]

    hsv_mean = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)

    if blue_green_in_bound(hsv_mean[0][0], low_green, high_green):
        return [0, 255, 0]
    if blue_green_in_bound(hsv_mean[0][0], low_blue, high_blue):
        return [255, 0, 0]
    if red_in_bound(hsv_mean[0][0], low_red, high_red, low_red_2, high_red_2):
        return [0, 0, 255]
    return [255, 255, 255]


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def modify_saturation(img, value):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] = hsv_img[..., 1] * value
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


def blue_green_in_bound(array, lower_bound, upper_bound):
    if array[0] >= lower_bound[0] and array[1] >= lower_bound[1] and array[2] >= lower_bound[2]:
        if array[0] <= upper_bound[0] and array[1] <= upper_bound[1] and array[2] <= upper_bound[2]:
            return True
    return False


def red_in_bound(array, lower_bound, upper_bound, red_second_lower_bound=None, red_second_upper_bound=None):
    if red_second_upper_bound is not None:
        if (array[0] >= lower_bound[0] and array[1] >= lower_bound[1] and array[2] >= lower_bound[2]) or \
                (array[0] >= red_second_lower_bound[0] and array[1] >= red_second_lower_bound[1] and array[2] >=
                 red_second_lower_bound[2]):
            if (array[0] <= upper_bound[0] and array[1] <= upper_bound[1] and array[2] <= upper_bound[2]) or \
                    (array[0] <= red_second_upper_bound[0] and array[1] <= red_second_upper_bound[1] and array[2] <=
                     red_second_upper_bound[2]):
                return True

    return False


def order_points(pts):
    most_left_top_to_bottom = sorted(sorted(pts, key=lambda element: (element[0][0], element[0][1]))[:2],
                                     key=lambda element: (element[0][1]))[:2]
    most_right_top_to_bottom = sorted(sorted(pts, key=lambda element: (element[0][0], element[0][1]))[2:4],
                                      key=lambda element: (element[0][1]))[:2]

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = most_left_top_to_bottom[0]  # Top left
    rect[1] = most_right_top_to_bottom[0]  # Top right
    rect[2] = most_left_top_to_bottom[1]  # Bottom left
    rect[3] = most_right_top_to_bottom[1]  # Bottom right
    return rect


def find_outer_bounds(img, border_buffer=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
        c += 1

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]

    # Get the corners of our grid
    corners = cv2.goodFeaturesToTrack(mask, 4, 0.1, 100)

    # Abort execution if no rectangle could be found
    if corners is None or len(corners) < 4:
        return False, _, _, _, _, _

    # Sort the corners
    most_left_top_to_bottom = sorted(sorted(corners, key=lambda element: (element[0][0], element[0][1]))[:2],
                                     key=lambda element: (element[0][1]))[:2]
    most_right_top_to_bottom = sorted(sorted(corners, key=lambda element: (element[0][0], element[0][1]))[2:4],
                                      key=lambda element: (element[0][1]))[:2]

    corners = deepcopy(corners)
    corners[0] = deepcopy(most_left_top_to_bottom[0])  # top left
    corners[1] = deepcopy(most_right_top_to_bottom[0])  # top right
    corners[2] = deepcopy(most_right_top_to_bottom[1])  # bottom right
    corners[3] = deepcopy(most_left_top_to_bottom[1])  # bottom left

    # Move corners slightly to create a buffer in order to improve grid detection
    corners[0][0][0] -= border_buffer
    corners[0][0][1] -= border_buffer
    corners[1][0][0] += border_buffer
    corners[1][0][1] -= border_buffer
    corners[2][0][0] += border_buffer
    corners[2][0][1] += border_buffer
    corners[3][0][0] -= border_buffer
    corners[3][0][1] += border_buffer

    # Create our warped image
    warped = four_point_transform(img, corners)
    cv2.imwrite('/home/test/Bildverarbeitung/warped.jpg', warped)
    return True, warped, corners[0], corners[1], corners[2], corners[3]


def find_grid(filename: str, resize: int, border_buffer: int):
    """Find a grid pattern in the given file.
    Returns a tuple containing a flattened image so the found grid is
    now a rectangle, and a `Grid` object representing the found grid.
    """
    img = cv2.imread(filename)
    img = imutils.resize(img, resize)
    found, new_img, top_left, top_right, bottom_right, bottom_left = find_outer_bounds(img, border_buffer)
    grid = Grid()
    return new_img, grid


def json(arguments):
    return _run(parse_args(arguments))


def _run(args):
    img, grid = find_grid(args.file, int(args.resize), int(args.bb))
    return grid_as_json(img, grid, args.hc, args.rc, float(args.saturation))


def run(arguments):
    _run(parse_args(arguments))


def _main():
    """Run from command line, parsing command line arguments"""
    _run(parse_args())


if __name__ == '__main__':
    _main()
