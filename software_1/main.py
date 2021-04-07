import sys
import time
import imutils
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

parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="input", help="Video file path")
parser.add_argument("-c_p", "--camera_port", dest="camera_port", help="Camera port, default 0 (WebCam)")

args = parser.parse_args()
input_file = vars(args)['input']
camera_port = vars(args)['camera_port']

    
if input_file is not None:
    if input_file == 'camera':
        # print(camera_port)
        port = 0 if camera_port is None else camera_port
        cap = cv2.VideoCapture(int(port))
    else:
        if os.path.isfile(input_file) and input_file.lower().endswith((".mp4", ".mkv")):
            cap = cv2.VideoCapture(input_file)
        else:
            sg.popup("Please select a valid video file!")
            exit(1)
else:
    config = yaml.safe_load(open('./config.yaml'))
    if config['input'] == 'browse':
        file_name = gui.browse_sample_video()
        cap = cv2.VideoCapture(file_name)
    elif config['input'] == 'camera':
        selected_camera_port = gui.show_list_of_cameras()
        cap = cv2.VideoCapture(selected_camera_port)

# config data for the program
max_pixel_movement = 5
max_displacement_per_frame = 2

# no of template
no_of_template = 3

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

# config variables
median_del_y = 0

template = None
red_line_dist = None

prev_sinked = -math.inf
prev_top_left = -math.inf

already_sinked = 0
sinked = 0

template_match_coord = []

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


# generate template's coordinates
def generate_templates(template_no):
    roi_h = abs(rect[1] - rect[3])
    offset_y = roi_h / template_no
    templates_coord = []
    
    for y in np.linspace(0, roi_h, template_no + 1).tolist()[:-1]:
        y1 = int(round(y)) + rect[1]
        y2 = int(round(y + offset_y)) + rect[1]
        templates_coord.append([(rect[0], y1), (rect[2], y2)])
    
    return templates_coord


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


# find best template's coordinates
def find_best_template(frame_to_get_templ, templ_coords):
    cnt_len_list = []
    
    for t_c in templ_coords:
        templ = frame_to_get_templ[t_c[0][1]:t_c[1][1], t_c[0][0]:t_c[1][0]]
        gray = cv2.bilateralFilter(templ, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnt_len_list.append(len(cnts))
    
    return np.argmax(cnt_len_list)


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


# writing to csv file
with open(data_filename, 'w', newline='', encoding='utf-8') as csv_file:
    # creating a csv writer object
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(data_fields)
    t = time.time()
    while True:
        ret, frame = cap.read()
        
        if ret is False:
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
            
            template_h = np.abs(rect[1] - rect[3]) / no_of_template
            if template is None or len(template_match_coord) == 20 or time.time() - t > 5:
                t = time.time()
                # print(f"resetting... {median_del_y}, {sinked}")
                # save data of the previous template
                if template is not None:
                    sinked = sinked + \
                            abs(template_match_coord[0] - template_match_coord[-1]) / red_line_dist
                # clear all the previous template's coordinates
                template_match_coord.clear()
                # generate template
                template_coord = generate_templates(no_of_template)
                templates_coord = divide_template_hor_ver(template_no_h=3, template_no_v=3)
                best_templ_idx = int(find_best_template(frame_gray, templates_coord))

                # select the best template
                best_template = templates_coord[best_templ_idx]
                
                # extract the template for visualizing purpose
                template = frame_gray[best_template[0][1]:best_template[1][1],
                           best_template[0][0]:best_template[1][0]]
                
                # reset top left coordinates
                prev_top_left = abs(rect[1] - best_template[0][1])
                
                # reset median_del_y for the current template
                median_del_y = abs(rect[1] - best_template[0][1])

            # print(len(np.unique(template)))
            # show the template
            cv2.imshow('template', template)
            
            # Perform the match operations
            res = cv2.matchTemplate(frame_gray[rect[1]:rect[3], rect[0]:rect[2]], template, cv2.TM_SQDIFF_NORMED)
            
            # find the template's location in the video
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            
            current_y = top_left[1]
            # find the next best match
            if abs(current_y - prev_top_left) > max_displacement_per_frame:
                match_result = np.array(res).flatten()
                sorted_match_result_idx = np.argsort(match_result)
                for idx in sorted_match_result_idx:
                    if idx >= prev_top_left:
                        # change the coord to next best match
                        current_y = idx
                        min_val = match_result[idx]
                        # print(f"changed Y-> {current_y}")
                        break
            
            if abs(current_y - prev_top_left) < max_pixel_movement:
                template_match_coord.append(current_y)
                # save the previous left Y coordinate to calculate noise in the data
                prev_top_left = current_y
                
            # print(current_y, prev_top_left)
            # track the median of the last 20 coordinate
            # this "median_del_y" is the grounded measurement for the current template
            # "median_del_y" also used in resetting the template
            # data updated interval = 20 frames
            # if len(template_match_coord) == 20:
            #     median_del_y = np.median(template_match_coord)
            #     template_match_coord.clear()
            #     sinked = already_sinked + (median_del_y / red_line_dist)
            
            # draw template match in the ROI
            w, h = template.shape[::-1]
            bottom_right = (top_left[0] + w, current_y + h)
            cv2.rectangle(frame[rect[1]:rect[3], rect[0]:rect[2]], (top_left[0], current_y), bottom_right, (255, 0, 0),
                          2)
            
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
        k = cv2.waitKey(18) & 0xff
        
        if k == ord('r'):
            release_resources()
            os.execl(sys.executable, sys.executable, *sys.argv)
        
        # if 'ESC' is pressed, quit the program
        if k == 27:
            # release resources
            release_resources()
            break
