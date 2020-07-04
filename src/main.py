
import cv2
import os
import time
import logging
import math
import numpy as np
from argparse import ArgumentParser
import sys
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection_model import FaceDetectionModel
from head_pose_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel
from face_landmark_model import FacialLandmarksDetectionModel

f_handler = logging.FileHandler('file.log')

logger = logging.getLogger('gaze_cursor_control')

logger.setLevel(logging.DEBUG)
logger.addHandler(f_handler)

benchmarks = {}

def build_arg_parser():

    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetection", required=True, type=str,
                        help="Path of Face Detection Model .xml file. required*")
    parser.add_argument("-fl", "--faciallandmark", required=True, type=str,
                        help="Path of Face Landmark Model .xml file. required*")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path of Head Pose Model .xml file. required*")
    parser.add_argument("-g", "--gazeestimation", required=True, type=str,
                        help="Path of Gaze Estimation model .xml file. required*")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Provide video path or write \"cam\" for camera streaming. required* ")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="Provide CPU extension")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="probability threshold for face detection values from 0.1 to 1")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Target device to be used by OpenVINO."
                             "CPU, GPU, FPGA or MYRIAD"
                             "Read https://docs.openvinotoolkit.org/latest/_docs_IE_DG_inference_engine_intro.html for more information."
                             "default: CPU")
    parser.add_argument("-pf","--previewFaceDetection",action="store_true",
                        help="Preview face detection")
    parser.add_argument("-pfl","--previewFaceLandmark",action="store_true",
                        help="Preview face landmarks")
    parser.add_argument("-php","--previewHeadPose",action="store_true",
                        help="Preview head pose")
    parser.add_argument("-pge","--previewGazeEstimation",action="store_true",
                        help="Preview gaze esitmation vector")

    parser.add_argument("-fliph", "--flip_horizontal", required=True, type=str,
                        default="False",
                        help="Flip input horizontally, incase of video is flipped horizontally")
    
    return parser

def is_model_exists(filename):
    if not os.path.isfile(filename):
            logger.error("Unable to find model ["+filename+"] xml file")
            return True
    return False

def increase_brightness(img, value=30):
    '''
    credit: https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def main():

    inputFilePath = args.input
    inputFeeder = None

    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)
    
    # Verify and Load Models
    face_model_link=args.facedetection
    facial_landmark_link=args.faciallandmark
    gaze_estimation_link=args.gazeestimation
    head_pose_link=args.headpose
    if is_model_exists(face_model_link) and is_model_exists(facial_landmark_link) and is_model_exists(gaze_estimation_link) and is_model_exists(head_pose_link):
        logger.log("Model not found! closing app...")
        exit(1)
    
    device_name = args.device
    cpu_extension = args.cpu_extension
    threshold = args.prob_threshold
    previewFace = args.previewFaceDetection
    previewFaceLandmark = args.previewFaceLandmark
    previewHeadPose = args.previewHeadPose
    previewGazeEstimation = args.previewGazeEstimation
    
    fliph = True if str(args.flip_horizontal).lower() == "true" else False

    # Initialize Models
    face_model=FaceDetectionModel(face_model_link,device_name,cpu_extension,threshold)
    facial_landmark_model=FacialLandmarksDetectionModel(facial_landmark_link,device_name,cpu_extension)
    gaze_estimation_model=GazeEstimationModel(gaze_estimation_link,device_name,cpu_extension)
    head_pose_model=HeadPoseEstimationModel(head_pose_link,device_name,cpu_extension)

    # Load Models
    fm_time = time.time()
    face_model.load_model()
    fm_time = time.time() - fm_time

    flm_time = time.time()
    facial_landmark_model.load_model()
    flm_time = time.time() - flm_time

    gem_time = time.time()
    gaze_estimation_model.load_model()
    gem_time = time.time() - gem_time

    hpm_time = time.time()
    head_pose_model.load_model()
    hpm_time = time.time() - hpm_time
    
    benchmarks['loadtime'] = {}
    benchmarks['loadtime']['face_landmark'] = flm_time
    benchmarks['loadtime']['face_detection'] = fm_time
    benchmarks['loadtime']['gaze_estimation'] = gem_time
    benchmarks['loadtime']['head_pose_estimation'] = hpm_time

    mouse_controller = MouseController('medium','medium')
    inputFeeder.load_data()

    frame_count = 0

    for ret, frame in inputFeeder.next_batch():
       
        if not ret:
            break

        # Waiting for 10ms for key input
        if cv2.waitKey(1) == 17:
            break

        FPS_COUNT = time.time()
        frame_count+=1
        increase_brightness(frame)
        if fliph:
            frame = cv2.flip(frame,1)

        face_detection_predict_time = time.time()    
        croppedFace, face_coords = face_model.predict(frame.copy())
        face_detection_predict_time = time.time()- face_detection_predict_time

        if croppedFace is None or croppedFace is 0:
            logger.error("Unable to detect the face.")            
            continue

        # Head Pose prediction

        head_pose_predict_time = time.time()
        head_output = head_pose_model.predict(croppedFace.copy())
        head_pose_predict_time = time.time() - head_pose_predict_time

        # Facial Landmark prediction

        facial_landmark_predict_time = time.time()
        left_eye, right_eye, eye_coords = facial_landmark_model.predict(croppedFace.copy())
        facial_landmark_predict_time = time.time() - facial_landmark_predict_time

        # Gaze Estimation prediction
        gaze_estimation_predict_time = time.time()
        gaze_vector,raw_vector = gaze_estimation_model.predict(left_eye, right_eye, head_output)
        gaze_estimation_predict_time = time.time() - gaze_estimation_predict_time

        FPS_COUNT = time.time()-FPS_COUNT
        FPS_COUNT = 1//FPS_COUNT
        logger.debug("FPS %s"%(FPS_COUNT))
        benchmarks['predict_time'] = {}
        benchmarks['predict_time']['face_landmark'] = facial_landmark_predict_time
        benchmarks['predict_time']['face_detection'] = face_detection_predict_time
        benchmarks['predict_time']['gaze_estimation'] = gaze_estimation_predict_time
        benchmarks['predict_time']['head_pose_estimation'] = head_pose_predict_time
        logger.debug(benchmarks)

        if previewFace or previewFaceLandmark or previewGazeEstimation or previewHeadPose:
            preview_frame = frame.copy()
            if previewFace:
                cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 1)
                preview_frame = croppedFace
            if previewFaceLandmark:
                cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 1)
                cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 1)

            if previewGazeEstimation and gaze_vector:
                x,y,z = gaze_vector
                x=int(x)
                y=int(y)
                # left eye center 
                left_eye_center_x = (eye_coords[0][0] + eye_coords[0][2])//2
                left_eye_center_y = (eye_coords[0][1] + eye_coords[0][3])//2
                left_eye_center_dx= left_eye_center_x+(x*2)          
                left_eye_center_dy= left_eye_center_y+(-y*2)  

                left_eye_ref = left_eye.copy()
                
                # right eye center 
                right_eye_center_x = (eye_coords[1][0] + eye_coords[1][2])//2
                right_eye_center_y = (eye_coords[1][1] + eye_coords[1][3])//2
                right_eye_center_dx= right_eye_center_x+(x*2)          
                right_eye_center_dy= right_eye_center_y+(-y*2)  

                right_eye_ref = right_eye.copy()
                              
                # head pose
                head_pose_y = head_output[1] * math.pi / 180

                line_size = 20
                
                # z-axis
                cv2.arrowedLine(croppedFace,(left_eye_center_x,left_eye_center_y),(left_eye_center_dx,left_eye_center_dy),(0,255,255),1)
                cv2.arrowedLine(croppedFace,(right_eye_center_x,right_eye_center_y),(right_eye_center_dx,right_eye_center_dy),(0,255,255),1)

                # x-axis
                cv2.arrowedLine(croppedFace,(left_eye_center_x,left_eye_center_y),(left_eye_center_x-line_size,left_eye_center_y),(0,0,255),1)
                cv2.arrowedLine(croppedFace,(right_eye_center_x,right_eye_center_y),(right_eye_center_x-line_size,right_eye_center_y),(0,0,255),1)
                
                # y-axis
                cv2.arrowedLine(croppedFace,(left_eye_center_x,left_eye_center_y),(left_eye_center_x,left_eye_center_y-line_size),(0,255,0),1)
                cv2.arrowedLine(croppedFace,(right_eye_center_x,right_eye_center_y),(right_eye_center_x,right_eye_center_y-line_size),(0,255,0),1)

            if previewHeadPose:
                cv2.rectangle(preview_frame,(5,5),(85,65),(0,255,0),1)
                cv2.putText(preview_frame,"YAW: {:.2f}".format(head_output[0]), (10, 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                cv2.putText(preview_frame,"PITCH: {:.2f}".format(head_output[1]), (10, 40), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                cv2.putText(preview_frame,"ROLL: {:.2f}".format(head_output[2]), (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

            cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
        
        if frame_count%5 == 0:
            logger.debug("moving mouse: {}".format(gaze_vector[0],gaze_vector[1]))
            mouse_controller.move(gaze_vector[0],gaze_vector[1])    
    logger.info("Video Stream Finished...")
    cv2.destroyAllWindows()
    inputFeeder.close()


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    try:
        main()
    except Exception as e:
        logger.error(e.with_traceback())