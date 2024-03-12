# Import required libraries
from ultralytics import YOLO
import cv2
import utilis
import math
import PIL
import streamlit as st
import numpy as np


# Replace the relative path to your weight file
model_path= "D:/New Folder (4)/YOLOv5/runs/detect/final train/weights/best.pt"
model= YOLO("D:/New Folder (4)/YOLOv5/runs/detect/final train/weights/best.pt")
def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    return is_display_tracker
def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    classNames = ["Fresh apple", "Rotten Apple", "Fresh Banana", "Rotten Banana", "Fresh Orange", "Rotten Orange"]
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
        boxes = res[0].boxes
        for box in boxes:

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
            print("Width of Box: {}, Height of Box: {}".format(x2, y2))
            # put box in cam
           # class name
            cls = int(box.cls[0])


            # Convert height and width according to reference object
            org = [x1, y2]
            org2 = [x1, x2]
            font = cv2.FONT_HERSHEY_DUPLEX
            fontScale = 0.6
            color = (255, 255, 0)
            thickness = 1
            height = int((y2 - y1)*0.03)
            width = int((x2 - x1)*0.03)
            print("Height -->", height)
            print("Width -->", width)

            # For class 'Fresh Apple'
            if cls == 0:
                sizerangeSmall = 5.5
                sizerangeMed = 7
                print(sizerangeSmall)
                if height < sizerangeSmall or width < sizerangeSmall:
                    size = "Small"
                elif height <= sizerangeMed or width <= sizerangeMed:
                    size = "Medium"
                else:
                    size = "Large"
                cv2.putText(image, size, (int(x2 - 130), int(y2 + 20)), font, fontScale, color, thickness)

            # For class 'Fresh Banana'
            if cls == 2:
                sizerangeSmall = 17
                sizerangeMed = 20
                if height < sizerangeSmall or width < sizerangeSmall:
                    size = "Small"
                elif height <= sizerangeMed or width <= sizerangeMed:
                    size = "Medium"
                else:
                    size = "Large"
                cv2.putText(image, size, (int(x2 - 130), int(y2 + 20)), font, fontScale, color, thickness)

            # For class 'Fresh Orange'
            if cls == 4:
                sizerangeSmall = 5
                sizerangeMed = 7
                if height < sizerangeSmall or width < sizerangeSmall:
                    size = "Small"
                elif height <= sizerangeMed or width <= sizerangeMed:
                    size = "Medium"
                else:
                    size = "Large"
                cv2.putText(image, size, (int(x2 - 130), int(y2 + 20)), font, fontScale, color, thickness)


            # cv2.putText(image, classNames[cls], (int(x2 - 130), int(y2 + 20)), font, fontScale, (0, 128, 0),
            #             thickness)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
# Setting page layout
st.set_page_config(
    page_title="Fruit Detection2",  # Setting page title
    page_icon="🤖",  # Setting page icon
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded"  # Expanding sidebar by default
)

# Choose Input Method
st.sidebar.header("Image/Webcam Config")
imgorvdo = st.sidebar.radio(
    "Select Task", ['Image Detection', 'Real-Time Detection'])

# Creating main page heading
st.title("Fruit Detection and Size Determination")
# Creating sidebar
if imgorvdo == 'Real-Time Detection':
    with st.sidebar:

        # Model Options
        confidence = float(st.slider(
            "Select Model Confidence", 25, 100, 40)) / 100



    vid_cap = cv2.VideoCapture(0)

    st_frame = st.empty()
    while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            vid_cap.set(3, 640)
            vid_cap.set(4, 480)
            if success:
                    _display_detected_frames(confidence,
                                             model,
                                             st_frame,
                                             image,

                                             )
            else:
                    vid_cap.release()
                    break

    st.sidebar.error("Error loading video: " )

if imgorvdo == 'Image Detection':
    with st.sidebar:
        st.header("Image")  # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        source_img = st.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        # Model Options
        confidence = float(st.slider(
            "Select Model Confidence", 25, 100, 40)) / 100

    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if source_img:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                     caption="Uploaded Image",
                     use_column_width= True
                     )

    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    if st.sidebar.button('Detect Fruit'):
        res = model.predict(uploaded_image,
                            conf=confidence
                            )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                     caption='Detected Fruit',
                     use_column_width= True
                     )
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("No image is uploaded yet!")

    if st.sidebar.button('Measure Dimension'):
        if source_img is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            img = cv2.rotate(opencv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #img = opencv_image
            scale = 2
            wP = 297 * scale
            hP = 210 * scale
            ###################################
            classNames = ["Fresh Apple", "Rotten Apple", "Fresh Banana", "Rotten Banana", "Fresh Orange", "Rotten Orange"]
            imgContours, conts = utilis.getContours(img, minArea=50000, filter=4)

            if len(conts) != 0:
                biggest = conts[0][2]
                # print(biggest)
                imgWarp = utilis.warpImg(img, biggest, wP, hP)
                imgContours2, conts2 = utilis.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)

                img1 = imgContours2
                res = model.predict(img1,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                for box in boxes:

                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                    print("Width of Box: {}, Height of Box: {}".format(x2, y2))
                    # put box in cam
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 255), 2)

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # Convert height and width according to reference object
                    org = [x1, y2]
                    org2 = [x1, x2]
                    font = cv2.FONT_HERSHEY_DUPLEX
                    fontScale = 0.6
                    color = (255, 0, 0)
                    thickness = 1
                    height = int((y2 - y1) / 20)
                    width= int((x2 - x1) / 20)
                    print("Height -->", height)
                    print("Width -->", width)

                    # For class 'Fresh Apple'
                    if cls == 0:
                        sizerangeSmall = 5.5
                        sizerangeMed = 7
                        print(sizerangeSmall)
                        if height < sizerangeSmall or width < sizerangeSmall:
                            size = "Small"
                        elif height <= sizerangeMed or width <= sizerangeMed:
                            size = "Medium"
                        else:
                            size = "Large"
                        cv2.putText(img1, size, (int(x1), int(y1 - 10)), font, fontScale, color, thickness)

                    # For class 'Fresh Banana'
                    if cls == 2:
                        sizerangeSmall = 17
                        sizerangeMed = 20
                        if height < sizerangeSmall or width < sizerangeSmall:
                            size = "Small"
                        elif height <= sizerangeMed or width <= sizerangeMed:
                            size = "Medium"
                        else:
                            size = "Large"
                        cv2.putText(img1, size, (int(x1), int(y1 - 10)), font, fontScale, color, thickness)

                    # For class 'Fresh Orange'
                    if cls == 4:
                        sizerangeSmall = 5
                        sizerangeMed = 7
                        if height < sizerangeSmall or width < sizerangeSmall:
                            size = "Small"
                        elif height <= sizerangeMed or width <= sizerangeMed:
                            size = "Medium"
                        else:
                            size = "Large"
                        cv2.putText(img1, size, (int(x1), int(y1 - 10)), font, fontScale, color, thickness)

                    wPout = format(wP)
                    cv2.putText(img1, classNames[cls], (int(x2 - 130), int(y2 + 20)), font, fontScale, (0, 128, 0),
                                thickness)
                # cv2.imshow('A4', img1)
                    with st.expander("Detection Results"):
                        st.write(width, height)
            with col2:
                # rotated_img = img.rotate(180)
                st.image(img1, caption="Measured Sizes", use_column_width=True, channels="BGR")

