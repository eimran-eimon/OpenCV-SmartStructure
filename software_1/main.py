import sys
import time
from collections import defaultdict, Counter
import yaml
import cv2
import numpy as np
from datetime import datetime
import os
import math
import csv
import gui
from argparse import ArgumentParser
import PySimpleGUI as sg
import sentry_sdk

sentry_sdk.init(
    "https://7eab204c7ac44e229cd1c332cf73dff5@o553112.ingest.sentry.io/5679913",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="input", help="Video file path")
parser.add_argument("-c_p", "--camera_port", dest="camera_port", help="Camera port, default 0 (WebCam)")

args = parser.parse_args()
input_file = vars(args)['input']
camera_port = vars(args)['camera_port']
config = yaml.safe_load(open('./config.yaml'))

if input_file is not None:
    if input_file == 'camera':
        port = 0 if camera_port is None else camera_port
        cap = cv2.VideoCapture(int(port))
    else:
        if os.path.isfile(input_file) and input_file.lower().endswith((".mp4", ".mkv")):
            cap = cv2.VideoCapture(input_file)
        else:
            sg.popup("Please select a valid video file!")
            exit(1)
else:
    if config['input'] == 'browse':
        file_name = gui.browse_sample_video()
        cap = cv2.VideoCapture(file_name)
    elif config['input'] == 'camera':
        selected_camera_port = gui.show_list_of_cameras()
        cap = cv2.VideoCapture(selected_camera_port)

# config data for the program
max_pixel_movement = 5
max_displacement_per_frame = 2

refresh_rate = 2  # 2s
prev_sinked = -math.inf
sinked = 0

# config variables
template_dict = {}
prev_top_left_dict = {}
template = None
red_line_dist = None
template_match_coord_dict = defaultdict(list)

# no of template
templ_no_h = 3
templ_no_v = 3
total_template_no = (templ_no_h - 1) * templ_no_v  # exclude the bottom one

# get random color color = {k: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for k in
# range(total_template_no)}

# ROI selection data
rect = (0, 0, 0, 0)
startPoint = False
endPoint = False
drawing = False

# csv field names
data_fields = ['Date-Time', 'Measurement (in ft)']
csv_directory = './stored_csv_files'

if not os.path.exists(csv_directory):
    os.makedirs(csv_directory)
# name of the csv file
data_filename = f"./{csv_directory}/data_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"

# FPS
start_time = time.time()
fps_show_interval = 1  # displays the frame rate every 1 second
counter = 0
fps = 0
frame_no = 1


# record on mouse events
def on_mouse(event, x, y, flags, params):
    global rect, startPoint, endPoint, drawing

    # get mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if not startPoint:
            rect = (x, y, 0, 0)
            startPoint = True
            drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and startPoint is True:
            rect = (rect[0], rect[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if startPoint and not endPoint:
            rect = (rect[0], rect[1], x, y)
            endPoint = True
            drawing = False


# divide and generate template's coordinates
def divide_template_hor_ver(template_no_h, template_no_v):
    roi_h = abs(rect[1] - rect[3])
    roi_v = abs(rect[0] - rect[2])

    offset_x = roi_v / template_no_v
    offset_y = roi_h / template_no_h

    templ_coord = []

    for x in np.linspace(0, roi_v, template_no_v + 1).tolist()[:-1]:
        x1 = int(round(x)) + rect[0]
        x2 = int(round(x + offset_x)) + rect[0]
        for y in np.linspace(0, roi_h, template_no_h + 1).tolist()[:-2]:
            y1 = int(round(y)) + rect[1]
            y2 = int(round(y + offset_y)) + rect[1]
            templ_coord.append([(x1, y1), (x2, y2)])

    return templ_coord


# release all the resources
def release_resources():
    cv2.destroyAllWindows()
    cap.release()
    csv_file.close()
    print("Process completed successfully!  All the data files could be found in the 'stored_csv_files' directory.")


# instruction texts
def put_instruction_texts():
    cv2.putText(frame, f"{instruction_text}",
                (50, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, instruction_color, 2)
    cv2.putText(frame, "Key r - Restart the program",
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (250, 0, 20), 2)
    cv2.putText(frame, "Key ESC - Quit the program",
                (50, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (250, 0, 20), 2)


def isTemplateMatchCoordLen(teml_match_coord_dict, length):
    for key in teml_match_coord_dict:
        if len(template_match_coord_dict[key]) < length:
            return False
    return True


def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    # print(top_two)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        # It is a tie, calc mean
        return np.mean([top_two[0][0], top_two[1][0]])
    return top_two[0][0]


# writing to csv file
with open(data_filename, 'w', newline='', encoding='utf-8') as csv_file:
    # creating a csv writer object
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(data_fields)
    t = time.time()

    while True:
        ret, frame = cap.read()
        input_fps = cap.get(cv2.CAP_PROP_FPS)

        if ret is False:
            if input_file == 'camera' or config['input'] == 'camera':
                sg.popup("Camera Disconnected! Please restart the program!")
            release_resources()
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mouse event's coordinates will be recorded from this window
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', on_mouse)

        # draw rectangle on mouse move
        if startPoint is True:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)

        if startPoint is True and endPoint is True:
            if red_line_dist is None:
                red_line_dist = np.abs(rect[1] - rect[3])
                # make start point and end point False to draw another rectangle
                startPoint = False
                endPoint = False
                continue

            if rect[1] > rect[3]:
                rect_list = list(rect)
                rect_list[1], rect_list[3] = rect_list[3], rect_list[1]
                rect = tuple(rect_list)
            if rect[0] > rect[2]:
                rect_list = list(rect)
                rect_list[0], rect_list[2] = rect_list[2], rect_list[0]
                rect = tuple(rect_list)

            # reset templates
            if not any(template_dict) or isTemplateMatchCoordLen(template_match_coord_dict,
                                                                 input_fps) or time.time() - t > refresh_rate:
                t = time.time()
                # save data of the previous template
                if template is not None:
                    displacement = []
                    for key in template_match_coord_dict:
                        match_coord = template_match_coord_dict[key]
                        displacement.append(abs(match_coord[0] - match_coord[-1]))

                    # most voted displacement
                    avg_displacement = find_majority(displacement)

                    sinked = sinked + (avg_displacement / red_line_dist)

                # clear all the previous template's coordinates
                template_match_coord_dict.clear()
                template_dict.clear()

                # generate templates
                templates_coord = divide_template_hor_ver(templ_no_h, templ_no_v)
                for i, template_coord in enumerate(templates_coord):
                    template_dict[i] = frame_gray[template_coord[0][1]:template_coord[1][1],
                                       template_coord[0][0]:template_coord[1][0]]
                    prev_top_left_dict[i] = abs(rect[1] - template_coord[0][1])

            # Perform the match operations
            for idx in template_dict:
                template = template_dict[idx]

                res = cv2.matchTemplate(frame_gray[rect[1]:rect[3], rect[0]:rect[2]], template, cv2.TM_SQDIFF_NORMED)

                # find the template's location in the video
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = min_loc

                current_y = top_left[1]
                # find the next best match
                if abs(current_y - prev_top_left_dict[idx]) > max_displacement_per_frame:
                    match_result = np.array(res).flatten()
                    sorted_match_result_idx = np.argsort(match_result)
                    for idx_match in sorted_match_result_idx:
                        if idx >= prev_top_left_dict[idx]:
                            # change the coord to next best match
                            current_y = idx_match
                            min_val = match_result[idx_match]
                            # print(f"changed Y-> {current_y}")
                            break

                # print(idx, current_y)
                if abs(current_y - prev_top_left_dict[idx]) < max_pixel_movement:
                    # print(template_match_coord_dict[idx])
                    template_match_coord_dict[idx].append(current_y)
                    # save the previous left Y coordinate to calculate noise in the data
                    prev_top_left_dict[idx] = current_y

                # draw template match in the ROI
                # w, h = template.shape[::-1]
                # bottom_right = (top_left[0] + w, current_y + h)
                # cv2.rectangle(frame[rect[1]:rect[3], rect[0]:rect[2]], (top_left[0], current_y), bottom_right,
                #               color[idx],
                #               2)

            cv2.putText(frame, "Sinked: {:.2f}ft".format(sinked), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # watch for change in measurement
            if sinked > prev_sinked:
                # write to the csv file
                csv_writer.writerows([[datetime.now().strftime('%Y-%m-%d %H:%M:%S'), round(sinked, 4)]])
                prev_sinked = sinked

        # change the instruction text and color according to the UX logic
        instruction_text = ""
        instruction_color = (0, 0, 0)

        if red_line_dist is None:
            instruction_text = "Please select 1ft area on the Piller"
            instruction_color = (0, 255, 0)
        elif template is None and red_line_dist > 0:
            instruction_text = "Please select the Region Of Interest"
            instruction_color = (0, 0, 255)

        # put instruction texts on the screen
        put_instruction_texts()

        # calc FPS
        counter += 1
        if (time.time() - start_time) > fps_show_interval:
            fps = counter / (time.time() - start_time)
            counter = 0
            start_time = time.time()

        cv2.putText(frame, f"Frame No: {frame_no}, FPS: {round(fps)}",
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)

        # show the resultant frame
        cv2.imshow('frame', frame)
        frame_no += 1
        # get the key input value
        k = cv2.waitKey(12) & 0xff

        if k == ord('r'):
            release_resources()
            os.execl(sys.executable, sys.executable, *sys.argv)

        # if 'ESC' is pressed, quit the program
        if k == 27:
            # release resources
            release_resources()
            break
