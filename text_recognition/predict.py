import argparse
import cv2
import itertools, os, time
import numpy as np
from keras import backend as K
from Model import get_Model
from parameter import letters

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import image_selection


K.set_learning_phase(0)

def labels_to_text(labels):     # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr


def crop_image(image, roi):
    #print roi
    retImg = image[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
    return retImg


def detect(image, roi_points):
    # conf = ('-l eng --oem 1 --psm 3')
    for i, selection in enumerate(range(int(len(roi_points)/2))):

        # sel = crop_image(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY),
        sel = crop_image(image.copy(),
                         [roi_points[selection*2],
                          roi_points[2*selection+1]])

        # sel = preprocess(sel)

        img_pred = sel.astype(np.float32)
        img_pred = cv2.resize(img_pred, (128, 64))
        img_pred = (img_pred / 255.0)# * 2.0 - 1.0
        img_pred = np.transpose(img_pred, (1,0,2))
        # img_pred = img_pred.T
        # img_pred = np.expand_dims(img_pred, axis=-1)
        img_pred = np.expand_dims(img_pred, axis=0)

        net_out_value = model.predict(img_pred)

        pred_texts = decode_label(net_out_value)
        result = labels_to_text(pred_texts)

        print("result", result)

        if i==0 or i==1:
            try:
                new_x = int(result)
            except:
                new_x = 0

            update_xy(i, new_x)

        cv2.rectangle(image, roi_points[selection*2],
                             roi_points[2*selection+1],
                             (0,255,0),
                             2)

        cv2.putText(image, result, roi_points[selection*2],
                                       cv2.FONT_HERSHEY_COMPLEX,
                                       1,
                                       (0,0,0),
                                       2)


x_max = 30
line1 = None
line2 = None
x1 = []
y1 = []
x2 = []
y2 = []
x_num = 0
fig = None


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,


def animate(i):

    global x1, y1, x2, y2, x_max

    if len(x1) > x_max:
        x1 = x1[:x_max]
        x2 = x2[:x_max]
        y1 = y1[1:x_max+1]
        y2 = y2[1:x_max+1]

    line1.set_data(x1, y1)
    line2.set_data(x2, y2)

    return line1, line2,


def update_xy(i, result):
    global x1, x2, y1, y2, x_num

    if i==0:
        x1.append(x_num)
        y1.append(float(result))

    elif i==1:
        x2.append(x_num)
        y2.append(float(result))


def init_graph():

    global fig
    global x1, y1, x2, y2
    global line1, line2

    plt.style.use('ggplot')
    # plt.style.use('seaborn-pastel')

    fig = plt.figure('Sensor Viewer')
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+%d+%d" % (1600, 400))

    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, 60)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Value')
    line1, = ax1.plot([], [], lw=2)
    line1.set_color('Red')

    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, 60)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Value')
    line2, = ax2.plot([], [], lw=2)
    line2.set_color('Blue')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", help="weight file directory",
                        type=str, default="best_model2.hdf5")
    args = parser.parse_args()

    # Get CRNN model
    model = get_Model(training=False)
    try:
        model.load_weights(args.weight)
        print("...Previous weight data...")
    except:
        raise Exception("No weight file!")


    init_graph()
    anim1 = FuncAnimation(fig, animate, init_func=init,
                                        frames=200,
                                        interval=20,
                                        repeat=True,
                                        blit=True)

    plt.ion()
    plt.show(False)

    h,w = 1280, 720







    # video file = 0
    # streaming = 1
    mode = 0
    if mode == 0:
        cap = cv2.VideoCapture("output.mp4")

    else:
        cap = cv2.VideoCapture("rtsp://admin:1234@ijoon.net:30084/h264")


    _, img = cap.read()
    img = cv2.resize(img, (h, w))
    selections = image_selection.getSelectionsFromImage(img)
    i = 0
    start_frame_index = 0

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        if mode == 0:
            idx = start_frame_index + i*30
            if length <= idx:
                print("done")
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            print(i)
            i += 20

        _, img = cap.read()

        if img is not None:
            img = cv2.resize(img, (h, w))
            detect(img, selections)

            cv2.imshow("result", img)

            x_num += 1
            plt.pause(0.001)
            plt.draw()

        if cv2.waitKey(1) == 27:
           break

    cv2.waitKey()
    cv2.destroyAllWindows()
