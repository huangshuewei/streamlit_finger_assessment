# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:43:36 2022

@author: Shu-wei Huang
"""

import streamlit as st
# from my_funs import getPrediction as gp

import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from keras import models
from skimage.transform import resize
import pandas as pd
from scipy import ndimage
import tempfile
import os
import base64

####
@st.cache(ttl=5)
def draw_hand_pts_lines(cv2_img,
                        handList,
                        thumb_color,
                        index_color,
                        middle_color,
                        ring_color,
                        pinky_color):
    
    # index
    index_end = np.array(handList[5])
    index_end[1]  = np.array(handList[0])[1]
    index_end = list(index_end)
    
    # middle
    middle_end = np.array(handList[9])
    middle_end[1]  = np.array(handList[0])[1]
    middle_end = list(middle_end)
    
    # ring
    ring_end = np.array(handList[13])
    ring_end[1]  = np.array(handList[0])[1]
    ring_end = list(ring_end)
    
    # pinky
    pinky_end = np.array(handList[17])
    pinky_end[1]  = np.array(handList[0])[1]
    pinky_end = list(pinky_end)
    
    cv2.circle(cv2_img, handList[5], 60, index_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[9], 60, middle_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[13], 60, ring_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[17], 60, pinky_color, cv2.FILLED)
    
    cv2.circle(cv2_img, handList[4], 60, thumb_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[8], 60, index_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[12], 60, middle_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[16], 60, ring_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[20], 60, pinky_color, cv2.FILLED)
    
    cv2.circle(cv2_img, handList[3], 20, thumb_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[7], 60, index_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[11], 60, middle_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[15], 60, ring_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[19], 60, pinky_color, cv2.FILLED)
    
    cv2.circle(cv2_img, handList[6], 60, index_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[10], 60, middle_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[14], 60, ring_color, cv2.FILLED)
    cv2.circle(cv2_img, handList[18], 50, pinky_color, cv2.FILLED)
    
    cv2.circle(cv2_img, index_end, 60, index_color, cv2.FILLED)
    cv2.circle(cv2_img, middle_end, 60, middle_color, cv2.FILLED)
    cv2.circle(cv2_img, ring_end, 60, ring_color, cv2.FILLED)
    cv2.circle(cv2_img, pinky_end, 60, pinky_color, cv2.FILLED)
    
    return cv2_img

@st.cache(ttl=5)
def getPrediction(hand_image):
    
    ## Set class names
    classes = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4','Grade 5', 'Grade 6']
    le = LabelEncoder()
    le.fit(classes)
    # le.inverse_transform([2])
    
    # Load model
    model = models.load_model("CNN_DenseNet169.h5",compile=False)
    
    # Apply mediapipe
    # First step is to initialize the Hands class an store it in a variable
    mp_hands = mp.solutions.hands
    
    # Now second step is to set the hands function which will hold the landmarks points
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    
    # Last step is to set up the drawing function of hands landmarks on the image
    # mp_drawing = mp.solutions.drawing_utils
        
    # Read image
    sample_img_color = hand_image
    sample_img_color = np.array(sample_img_color)
    sample_img_color = resize(sample_img_color, [int(np.shape(sample_img_color)[0]/np.shape(sample_img_color)[0]*1920), 
                                                 int(np.shape(sample_img_color)[1]/np.shape(sample_img_color)[0]*1920), 
                                                 3])*sample_img_color.max()
    sample_img_color = sample_img_color.astype('uint8')
    
    sample_img = hand_image
    
    sample_img = np.array(sample_img)
    
    # Crop finger imagers in the frontal-hand image
    # 1st. process the hand image
    results = hands.process(sample_img[:,:,0:3])
    
    plain_img = np.zeros((int(np.shape(sample_img)[0]/np.shape(sample_img)[0]*1920), 
                          int(np.shape(sample_img)[1]/np.shape(sample_img)[0]*1920), 
                          3))
    handList = []
    
    # 2ed. Extract the coordinate of joints
    if results.multi_hand_landmarks:
            
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, c = plain_img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handList.append((cx, cy))
    
    # 3rd. Draw the points of joints on the hand image
    plain_img = draw_hand_pts_lines(plain_img,
                                    handList,
                                   [54, 54, 54],
                                [104, 104, 104],
                                [154, 154, 154],
                                [204, 204, 204],
                                [255, 255, 255])
    rgb_weights = [0.1140, 0.2989, 0.5870]
    grayscale_image = np.dot(plain_img[...,:3], rgb_weights)
        
    sample_img = resize(sample_img, [int(np.shape(sample_img)[0]/np.shape(sample_img)[0]*1920), 
                                     int(np.shape(sample_img)[1]/np.shape(sample_img)[0]*1920), 
                                     3])*sample_img.max()
    sample_img = sample_img.astype('uint8')

    # Label of joints' locations
    le2 = LabelEncoder()
    h, w = grayscale_image.shape
    grayscale_image_reshaped = grayscale_image.reshape(-1,1)
    grayscale_image_reshaped_encoded = le2.fit_transform(grayscale_image_reshaped)
    grayscale_image_reshaped_encoded_original_shape = grayscale_image_reshaped_encoded.reshape(h, w)
    
    # Start cropping finger images
    # index
    index_mask = np.where(grayscale_image_reshaped_encoded_original_shape == 2)
    index_area = sample_img[index_mask[0].min():index_mask[0].max(),index_mask[1].min():index_mask[1].max()]
    index_area = resize(index_area, [512, 128, 3])
    
    # midl
    midl_mask = np.where(grayscale_image_reshaped_encoded_original_shape == 3)
    midl_area = sample_img[midl_mask[0].min():midl_mask[0].max(),midl_mask[1].min():midl_mask[1].max()]
    midl_area = resize(midl_area, [512, 128, 3])
    
    # ring
    ring_mask = np.where(grayscale_image_reshaped_encoded_original_shape == 4)
    ring_area = sample_img[ring_mask[0].min():ring_mask[0].max(),ring_mask[1].min():ring_mask[1].max()]
    ring_area = resize(ring_area, [512, 128, 3])
    
    # picky
    picky_mask = np.where(grayscale_image_reshaped_encoded_original_shape == 5)
    picky_area = sample_img[picky_mask[0].min():picky_mask[0].max(),picky_mask[1].min():picky_mask[1].max()]
    picky_area = resize(picky_area, [512, 128, 3])
    
    # Draw fingers areas on hand image
    sample_img_color = cv2.rectangle(sample_img_color,
                                     (index_mask[1].min(),index_mask[0].min()),
                                     (index_mask[1].max(),index_mask[0].max()),
                                     (255,0,0),10)
    sample_img_color = cv2.rectangle(sample_img_color,
                                     (midl_mask[1].min(),midl_mask[0].min()),
                                     (midl_mask[1].max(),midl_mask[0].max()),
                                     (255,255,0),10)
    sample_img_color = cv2.rectangle(sample_img_color,
                                     (ring_mask[1].min(),ring_mask[0].min()),
                                     (ring_mask[1].max(),ring_mask[0].max()),
                                     (0,255,0),10)
    sample_img_color = cv2.rectangle(sample_img_color,
                                     (picky_mask[1].min(),picky_mask[0].min()),
                                     (picky_mask[1].max(),picky_mask[0].max()),
                                     (0,0,255),10)
    
    # Do prediction
    # Combine all finger images
    input_fingers = []
    input_fingers.append(index_area)
    input_fingers.append(midl_area)
    input_fingers.append(ring_area)
    input_fingers.append(picky_area)
    input_fingers = np.array(input_fingers)
    
    # Predicting
    grades = model.predict(input_fingers)
    grades = np.argmax(grades,axis = 1) 
    grades = le.inverse_transform(grades)
    
    # plt.imshow(sample_img_color)
    
    out_put = ("Index finger: " + grades[0] + ",Middle finger: " + grades[1] + ",Ring finger: " + grades[2] + ",Little finger: " + grades[3]).split(',')
    sample_img_color = cv2.putText(sample_img_color, out_put[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   2, (0, 0, 0), 5, cv2.LINE_AA)
    sample_img_color = cv2.putText(sample_img_color, out_put[1], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                   2, (0, 0, 0), 5, cv2.LINE_AA)  
    sample_img_color = cv2.putText(sample_img_color, out_put[2], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                   2, (0, 0, 0), 5, cv2.LINE_AA)
    sample_img_color = cv2.putText(sample_img_color, out_put[3], (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                                   2, (0, 0, 0), 5, cv2.LINE_AA)
    
    return out_put, grades, sample_img_color
####

logo_img = Image.open("logo.png")
st.image(logo_img)  
st.write("The digital Manchester Digit Score (dMDS)")
st.write("This is a prototype which is used to assess finger function.")
st.write("Methods are based on machine learning and computer vision.")
load_img = None

c1, c2, c3 = st.columns(3)


with c1:
    st.title("Upload a frontal-hand image.")
    
    uploaded_file = st.file_uploader("Choose a frontal-hand image taken by smartphone...")
    
    if uploaded_file:
        load_img = Image.open(uploaded_file)
        load_img = np.array(load_img)
        img_shape = np.shape(load_img)
        
        if img_shape[1] > img_shape[0]:
            load_img = ndimage.rotate(load_img, 180, reshape=True)
            load_img = ndimage.rotate(load_img, 90, reshape=True)
        
        print("check point")
        st.image(load_img)   

    with c2:
        st.title("Analysing after uploading the image.")
        if np.array(load_img) is not None:
            if st.button('Start analysing'):
                __, assessed_result, assessed_img = getPrediction(load_img)
                df = pd.DataFrame({'Fingers': ["Index", "Middle", "Ring", "Little"],
                                   'Grades': [assessed_result[0], assessed_result[1], 
                                              assessed_result[2], assessed_result[3]]})
                st.title("Result:")
                st.table(df)
                st.image(assessed_img)


with c3:
    st.title("Not open yet.")
    uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])

    if uploaded_video is not None: # run only when user uploads video
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        temp_dir = tempfile.mkdtemp()
        temp_output_path = os.path.join(temp_dir, "output.mp4")

        st.markdown(f"""
        ### Files
        - {vid}
        """,
        unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk

        # Get video details
        fps = int(vidcap.get(5))
        frame_skip = 2*fps

        cur_frame = 0
        success = True

        # Define codec and create VideoWriter object to save the processed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, 0.5, (1080, 1920), isColor=False)

        set_images = []

        # Process
        while success:
            success, frame = vidcap.read() # get next frame from video
            
            if cur_frame % frame_skip == 0: # only analyze every n=300 frames
                print('frame: {}'.format(cur_frame)) 
                __, assessed_result, assessed_img = getPrediction(frame)
                assessed_img[:,:,[0,1,2]] = assessed_img[:,:,[2,1,0]]
                set_images.append(assessed_img)
            cur_frame += 1
        
        set_images = np.array(set_images)

        set_images.shape

        if set_images is not []:

            # Write the processed frame to the output video
            for i in range(len(set_images)):
                out.write(set_images[i])
        vidcap.release()
        out.release()

        st.video(temp_output_path)

        # Offer download link for processed video
        with open(temp_output_path, "rb") as video_file:
            video_bytes = video_file.read()
        video_b64 = base64.b64encode(video_bytes).decode()
        st.download_button("Download Processed Video", video_b64, key="download_button")

            # st.video(out)
            # st.image(set_images)