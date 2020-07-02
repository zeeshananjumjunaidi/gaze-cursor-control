
import cv2
import os
import time
import logging
import numpy as np
from argparse import ArgumentParser

from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection_model import FaceDetectionModel
from head_pose_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel
from face_landmark_model import FacialLandmarksDetectionModel


logger = logging.getLogger()

benchmarks = {}

def build_arg_parser():

    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")

    parser.add_argument("-fliph", "--flip_horizontal", required=True, type=str,
                        default="False",
                        help="Flip input horizontally")
    
    return parser

def is_model_exists(filename):
    if not os.path.isfile(filename):
            logger.error("Unable to find model ["+filename+"] xml file")
            return True
    return False

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
    face_model_link=args.facedetectionmodel
    facial_landmark_link=args.faciallandmarkmodel
    gaze_estimation_link=args.gazeestimationmodel
    head_pose_link=args.headposemodel
    if is_model_exists(face_model_link) and is_model_exists(facial_landmark_link) and is_model_exists(gaze_estimation_link) and is_model_exists(head_pose_link):
        logger.log("Model not found! closing app...")
        exit(1)
    
    device_name = args.device
    cpu_extension = args.cpu_extension
    threshold = args.prob_threshold
    previewFlags = args.previewFlags
    
    fliph = True if str(args.flip_horizontal).lower() == "true" else False

    # Initialize Models
    face_model=FaceDetectionModel(face_model_link,device_name,cpu_extension)
    facial_landmark_model=FacialLandmarksDetectionModel(facial_landmark_link,device_name,cpu_extension)
    gaze_estimation_model=GazeEstimationModel(gaze_estimation_link,device_name,cpu_extension)
    head_pose_model=HeadPoseEstimationModel(head_pose_link,device_name,cpu_extension)

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


    mouse_controller = MouseController('medium','fast')
    inputFeeder.load_data()

    frame_count = 0

    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1

        # if frame_count%5==0:
        #     cv2.imshow('video',cv2.resize(frame,(500,500)))

        key = cv2.waitKey(60)
        if fliph:
            frame = cv2.flip(frame,1)
        face_detection_predict_time = time.time()    
        croppedFace, face_coords = face_model.predict(frame.copy())
        face_detection_predict_time = time.time()- face_detection_predict_time

        if type(croppedFace)==int:# or type(croppedFace)==np.float32:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue

        # Head Pose prediction

        head_pose_predict_time = time.time()
        head_output = head_pose_model.predict(croppedFace.copy())
        head_pose_predict_time = time.time() - head_pose_predict_time
        #print(head_output)
        # Eyes and its coordinate prediction
        facial_landmark_predict_time = time.time()
        left_eye, right_eye, eye_coords = facial_landmark_model.predict(croppedFace.copy())
        facial_landmark_predict_time = time.time() - facial_landmark_predict_time

        #print(left_eye,right_eye,eye_coords)
        using_only_head_movement=False
        if using_only_head_movement:
            new_mouse_coord = -head_output[0] *0.5,-head_output[1]*0.5
            gaze_vector=None
            gaze_estimation_predict_time = -1
        else:
            # Gaze Estimation prediction
            gaze_estimation_predict_time = time.time()
            new_mouse_coord, gaze_vector = gaze_estimation_model.predict(left_eye, right_eye, head_output)
            gaze_estimation_predict_time = time.time() - gaze_estimation_predict_time

        benchmarks['predict_time'] = {}
        benchmarks['predict_time']['face_landmark'] = facial_landmark_predict_time
        benchmarks['predict_time']['face_detection'] = face_detection_predict_time
        benchmarks['predict_time']['gaze_estimation'] = gaze_estimation_predict_time
        benchmarks['predict_time']['head_pose_estimation'] = head_pose_predict_time
        print(benchmarks)
        if (not len(previewFlags)==0):
            preview_frame = frame.copy()
            #print('fd' in previewFlags,'hp' in previewFlags,'fld' in previewFlags )
            if 'fd' in previewFlags:
                cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 2)
                preview_frame = croppedFace
            if 'fld' in previewFlags:
                cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 2)
                cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 2)
                #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                
            if 'hp' in previewFlags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(head_output[0],head_output[1],head_output[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            if 'ge' in previewFlags and gaze_vector:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                #preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = croppedFace
                
            cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
        
        if frame_count%5==0:
            print("moving mouse")
        mouse_controller.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
                break
    logger.log("Video Stream Finished...")
    cv2.destroyAllWindows()
    inputFeeder.close()


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    main()