"""
image_selection.py
This module gives the user an image to specify selections
in and returns those selections
@author: Suyash Kumar <suyashkumar2003@gmail.com>
"""
import cv2
import sys

# Shared variable declarations
refPts = []
image=1
numSelected=0
drawing = False

"""
Called every time a click event is fired on the displayed
image. Is used to record ROI selections from the user, and
updates the image to place a bounding box highlighting the
ROIs
"""
def click_handler(event, x, y, flags, pram):
    global refPts, image, numSelected, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        if(len(refPts)==0):
            refPts = [(x,y)]
        else:
            refPts.append((x,y))
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE: # 마우스 이동
        if drawing:
            copy = image.copy()
            cv2.rectangle(copy, refPts[numSelected*2], (x,y), (0,255,0), 2, lineType=8)
            cv2.imshow("camera",copy)

    elif (event == cv2.EVENT_LBUTTONUP):
        drawing = False
        refPts.append((x,y))

        x_min = min(refPts[numSelected*2][0], refPts[(numSelected*2)+1][0])
        x_max = max(refPts[numSelected*2][0], refPts[(numSelected*2)+1][0])

        y_min = min(refPts[numSelected*2][1], refPts[(numSelected*2)+1][1])
        y_max = max(refPts[numSelected*2][1], refPts[(numSelected*2)+1][1])

        refPts[numSelected*2] = (x_min, y_min)
        refPts[(numSelected*2)+1] = (x_max, y_max)

        cv2.rectangle(image, refPts[numSelected*2], refPts[(numSelected*2)+1], (0,255,0), 2, lineType=8)
        cv2.imshow("camera",image)
        numSelected = numSelected+1

def getSelectionsFromImage(img):
    global image, refPts, numSelected
    image = img
    refPts = [] # Reinit refPts
    clone = image.copy()
    cv2.namedWindow("camera")
    cv2.setMouseCallback("camera", click_handler)
    cv2.imshow("camera", image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if (key == ord('\r')):
            break

        if (key == 27):
            numSelected = max(0, numSelected-1)
            refPts = refPts[:numSelected*2]
            image = clone.copy()

            for i in range(numSelected):
                cv2.rectangle(image, refPts[i*2], refPts[(i*2)+1], (0,255,0), 2, lineType=8)
            cv2.imshow("camera",image)

    if ((len(refPts) % 2) == 0):
        print(numSelected)
        print(refPts)
        for i in range(numSelected):
            roi = clone[refPts[0+(i*2)][1]:refPts[1+(i*2)][1],
                        refPts[0+(2*i)][0]:refPts[1+(2*i)][0]]
            # cv2.imshow("ROI"+str(i), roi)
    else:
        sys.exit("Selection Capture didn't get an even number of bounding points.")

    cv2.waitKey(13)
    cv2.destroyAllWindows()

    return refPts
