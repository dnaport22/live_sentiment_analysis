import numpy as np
import numpy

# Imports the Google Cloud client librar

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from PIL import Image
import collections
import cv2

class ObjectSlicer():

    STANDARD_COLORS = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]

    def __init__(self,
                 image,
                 boxes,
                 classes,
                 scores,
                 category_index,
                 instance_masks=None,
                 keypoints=None,
                 use_normalized_coordinates=False,
                 max_boxes_to_draw=20,
                 min_score_thresh=.5,
                 agnostic_mode=False,
                 line_thickness=4):

        self.image = image
        self.boxes = boxes
        self.classes = classes
        self.scores = scores
        self.category_index = category_index
        self.instance_masks = instance_masks
        self.keypoints = keypoints
        self.use_normalized_coordinates = use_normalized_coordinates
        self.min_score_thresh = min_score_thresh
        self.agnostic_mode = agnostic_mode
        self.line_thickness = line_thickness
        self.max_boxes_to_draw = max_boxes_to_draw
        self.image_to_analyse = None

        if not max_boxes_to_draw:
            self.max_boxes_to_draw = boxes.shape[0]

    def slice_object(self):
        self.__box_map()
        # Draw all boxes onto image.
        for box, color in self.box_to_color_map.items():
            ymin, xmin, ymax, xmax = box

            self.draw_bounding_box_on_image_array(
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=self.line_thickness,
                display_str_list=self.box_to_display_str_map[box],
                use_normalized_coordinates=self.use_normalized_coordinates)

    def __box_map(self):
        """Overlay labeled boxes on an image with formatted scores and label names.
        This function groups boxes that correspond to the same location
        and creates a display string for each detection and overlays these
        on the image. Note that this function modifies the image in place, and returns
        that same image.
        Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            boxes: a numpy array of shape [N, 4]
            classes: a numpy array of shape [N]. Note that class indices are 1-based,
                and match the keys in the label map.
            scores: a numpy array of shape [N] or None.  If scores=None, then
                this function assumes that the boxes to be plotted are groundtruth
                boxes and plot all boxes as black with no classes or scores.
            category_index: a dict containing category dictionaries (each holding
                category index `id` and category name `name`) keyed by category indices.
            instance_masks: a numpy array of shape [N, image_height, image_width], can
                be None
            keypoints: a numpy array of shape [N, num_keypoints, 2], can
                be None
            use_normalized_coordinates: whether boxes is to be interpreted as
                normalized coordinates or not.
            max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
                all boxes.
            min_score_thresh: minimum score threshold for a box to be visualized
            agnostic_mode: boolean (default: False) controlling whether to evaluate in
                class-agnostic mode or not.  This mode will display scores but ignore
                classes.
            line_thickness: integer (default: 4) controlling line width of the boxes.
        Returns:
            uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
        """
        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        self.box_to_display_str_map = collections.defaultdict(list)
        self.box_to_color_map = collections.defaultdict(str)
        self.box_to_instance_masks_map = {}
        self.box_to_keypoints_map = collections.defaultdict(list)
        if not self.max_boxes_to_draw:
            self.max_boxes_to_draw = self.boxes.shape[0]
        for i in range(min(self.max_boxes_to_draw, self.boxes.shape[0])):
            if self.scores is None or self.scores[i] > self.min_score_thresh:
                box = tuple(self.boxes[i].tolist())
                if self.instance_masks is not None:
                    self.box_to_instance_masks_map[box] = self.instance_masks[i]
                if self.keypoints is not None:
                    self.box_to_keypoints_map[box].extend(self.keypoints[i])
                if self.scores is None:
                    self.box_to_color_map[box] = 'black'
                else:
                    if not self.agnostic_mode:
                        if self.classes[i] in self.category_index.keys():
                            class_name = self.category_index[self.classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = '{}: {}%'.format(
                            class_name,
                            int(100*self.scores[i]))
                    else:
                        display_str = 'score: {}%'.format(int(100 * self.scores[i]))
                    self.box_to_display_str_map[box].append(display_str)
                    if self.agnostic_mode:
                        self.box_to_color_map[box] = 'DarkOrange'
                    else:
                        self.box_to_color_map[box] = ObjectSlicer.STANDARD_COLORS[
                            self.classes[i] % len(ObjectSlicer.STANDARD_COLORS)]

    def draw_bounding_box_on_image_array(self,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
        """Adds a bounding box to an image (numpy array).
            Args:
                image: a numpy array with shape [height, width, 3].
                ymin: ymin of bounding box in normalized coordinates (same below).
                xmin: xmin of bounding box.
                ymax: ymax of bounding box.
                xmax: xmax of bounding box.
                color: color to draw bounding box. Default is red.
                thickness: line thickness. Default value is 4.
                display_str_list: list of strings to display in box
                                  (each to be shown on its own line).
                use_normalized_coordinates: If True (default), treat coordinates
                  ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
                  coordinates as absolute.
            """
        image_pil = Image.fromarray(np.uint8(self.image)).convert('RGB')
        self.draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                        thickness, display_str_list,
                                        use_normalized_coordinates)
        np.copyto(self.image, np.array(image_pil))

    def draw_bounding_box_on_image(self, image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):

        """Adds a bounding box to an image.
            Each string in display_str_list is displayed on a separate line above the
            bounding box in black text on a rectangle filled with the input 'color'.
            If the top of the bounding box extends to the edge of the image, the strings
            are displayed below the bounding box.
            Args:
                image: a PIL.Image object.
                ymin: ymin of bounding box.
                xmin: xmin of bounding box.
                ymax: ymax of bounding box.
                xmax: xmax of bounding box.
                color: color to draw bounding box. Default is red.
                thickness: line thickness. Default value is 4.
                display_str_list: list of strings to display in box
                                  (each to be shown on its own line).
                use_normalized_coordinates: If True (default), treat coordinates
                  ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
                  coordinates as absolute.
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)

            if display_str.split(':')[0] == 'person':
                img = image.crop((left, top, right, bottom))
                img_str = cv2.cvtColor(numpy.array(img), cv2.COLOR_BGR2GRAY)

                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', img_str, encode_param)
                data = numpy.array(imgencode)
                self.image_to_analyse = data.tostring()

            # draw.line([(left, top), (left, bottom), (right, bottom),
            #     (right, top), (left, top)], width=thickness, fill=color)
            #
            # draw.rectangle(
            #     [(left, text_bottom - text_height - 2 * margin), (left + text_width,
            #                                             text_bottom)],
            #     fill=color)
            #
            # draw.text(
            #     (left + margin, text_bottom - text_height - margin),
            #         display_str,
            #         fill='black',
            #         font=font)
            #
            # text_bottom -= text_height - 2 * margin

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)