# img_viewer.py

import PySimpleGUI as sg
import os.path


def browse_sample_video():
    browse_file = [
        [
            sg.Text("Sample Video:"),
            sg.In(size=(50, 1), enable_events=True, key="-FILE-"),
            sg.FileBrowse(),
        ]
    ]

    layout = [
        [
            sg.Column(browse_file)
        ]
    ]

    window = sg.Window("Video File Input", layout)

    # Run the Event Loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "-FILE-":
            file_name = values["-FILE-"]
            print(file_name)
            if os.path.isfile(file_name) and file_name.lower().endswith((".mp4", ".mkv")):
                window.close()
                return file_name
            else:
                sg.popup("Please select a valid file!")


