from camera_parameters import * 
import torch
import numpy as np
import cv2, os, time, datetime
import random
import detectron2
from bt_server import BTServer
from predictor import MTSDPredictor
from detection_utils import *
from detectron2.config import get_cfg
import threading
import statistics



def gstreamer_pipeline(
    capture_width=CAP_WIDTH,
    capture_height=CAP_HEIGHT,
    display_width=CAP_WIDTH,
    display_height=CAP_HEIGHT,
    framerate=FPS,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class CamVideoWriter:
    '''
    class for writing the video from the cam
    '''
    def __init__(self, out):
        self.out = out
        self.next_id=0
        self.curr_id=0

    def write(self, frames):
        t = threading.Thread(target=self._write_frames, args=[frames])
        t.daemon = True
        t.start()
        return self
    
    def _write_frames(self, frames):
        thread_id = self.curr_id
        self.curr_id = self.curr_id +1
        while(thread_id !=self.next_id):
            time.sleep(0.1)
        for f in frames:
            self.out.write(f)
            #print("write")
        self.next_id = self.next_id + 1


class InferenceVideoWriter:
    '''
    class for writing the video of inference
    '''
    def __init__(self, out):
        self.out = out

    def write(self, frame, fps):
        self.frame = frame
        self.fps = fps
        self.writing = True
        t = threading.Thread(target=self._write_frames)
        t.daemon = True
        t.start()
        return self
    
    def _write_frames(self):
        while self.writing:
            self.out.write(self.frame)
            time.sleep((1/self.fps)* 0.66)



class VideoCapture:
    '''
    class for reading frames from the camera. It keeps the most recent frame.
    '''
    def __init__(self, cap):
        self.cap = cap

    def startReadingFrames(self):
        _, self.latest_frame = self.cap.read()
        t = threading.Thread(target=(self._read_frames))
        t.daemon = True
        t.start()
        return self

    def _read_frames(self):
        self.frame_num = 0
        read = True
        while self.cap.isOpened():
            self.frame_num=self.frame_num+1
            #self.cap.grab()
            #if(i%2==0):
            #    read, self.latest_frame = self.cap.retrieve()
            read, self.latest_frame = self.cap.read()
            time.sleep(0.118)

    
    def read(self):
       return self.latest_frame

    def release(self):
        self.cap.release()



def run_camera_inference(bt_server, predictor):
    '''
    method for running the inference on a live camera image
    '''
    cap =VideoCapture(cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER))
    predictor(np.zeros((CROPPED_WIDTH, CROPPED_HEIGHT,3)))
    predictor(np.zeros((CROPPED_WIDTH, CROPPED_HEIGHT,3)))
    
    if cap.cap.isOpened():
        cap.startReadingFrames()
        try:
            while True:
                tic = time.time()
                img = cap.read()
                cropped_im = crop_img(img)
                visualisation = cropped_im.copy()
                if(bt_server.detecting):
                    outputs = predictor(cv2.resize(cropped_im, (CROPPED_WIDTH, CROPPED_HEIGHT)))
                    if(len(outputs['instances'])> 0):
                        bt_server.startSendingDetections(outputs, cropped_im)
                        # boxes, classes, scores = handle_prediction(outputs)
                        # boxes, classes, scores = nms(boxes, classes, scores, 0.5)
                        # for i, box in enumerate(boxes):
                        #     box_top = max(0, box[1] * SCALE_RATIO)
                        #     box_bottom = min(CAP_HEIGHT, box[3] * SCALE_RATIO)
                        #     box_l = max(0, box[0] * SCALE_RATIO)
                        #     box_r = min(CAP_WIDTH, box[2] * SCALE_RATIO)
                        #     visualisation = cv2.rectangle(
                        #         visualisation, 
                        #         (int(box_l), int(box_top)), 
                        #         (int(box_r), int(box_bottom)),
                        #         (0,255,0), 
                        #         2
                        #     ) 

                        #     visualisation = cv2.putText(
                        #         visualisation,
                        #         f"{classes[i]}, {scores[i]:.2f}",
                        #         (int(box_l),int(box_top*0.95)),
                        #         cv2.FONT_HERSHEY_SIMPLEX,
                        #         0.4,
                        #         (0,255,0),
                        #         1
                        #     )
                    print("detection time", time.time() - tic)
                #cv2.imshow("Camera", visualisation)

                if(bt_server.test_required):
                    bt_server.sendTestImage(img)
                
                try:
                    bt_server.socket.getpeername()
                except:
                    bt_server.detecting = False
                    bt_server.startAccepting()

                keyCode = cv2.waitKey(30) & 0xFF
                if keyCode == 27:
                    break

            cap.cap.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("stopped")
            cap.cap.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


def run_video_inference(predictor, filename):
    '''
    method for running the inference on a video file
    '''
    cap = VideoCapture(cv2.VideoCapture(filename))
    name = "output{}".format(datetime.datetime.now().time())
    #_out = cv2.VideoWriter('/home/pkos/workspace/praca_inzynierska/jetson_app/inference_out/{}.mp4'.format(name), 
    #    cv2.VideoWriter_fourcc(*'MP4V'), 8, (CAP_WIDTH,CAP_HEIGHT))
    #out = InferenceVideoWriter(_out)
    predictor(np.zeros((CROPPED_WIDTH, CROPPED_HEIGHT,3)))
    predictor(np.ones((CROPPED_WIDTH, CROPPED_HEIGHT,3)))
    
    if cap.cap.isOpened():
        #window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
        cap.startReadingFrames()
        fpss = []
        detections=[]
        starttime = time.time()
        #writing_started = False
        videostarttime=None
        try:
            tic = time.time()
            while True:
                detection = False
                img = cap.read()
                if(img is None):
                    break
                #visualisation = img
                cropped_im = crop_img(img)
                #cropped_im = img
                visualisation = img#cropped_im#.copy()

                outputs = predictor(cv2.resize(cropped_im, (CROPPED_WIDTH, CROPPED_HEIGHT)))
                if(len(outputs['instances'])> 0):
                    detection=True
                    #print("detection")
                    #bt_server.startSendingDetections(outputs, cropped_im)
                    boxes, classes, scores = handle_prediction(outputs)
                    boxes, classes, scores = nms(boxes, classes, scores, 0.5)
                    for i, box in enumerate(boxes):
                        detections.append((int(time.time() - starttime), f"{classes[i]}, {scores[i]:.2f}", cap.frame_num))
                        box_top = max(0, box[1] * SCALE_RATIO + 0.1 * CAP_HEIGHT)
                        box_bottom = min(CAP_HEIGHT, box[3] * SCALE_RATIO + 0.1 * CAP_HEIGHT)
                        box_l = max(0, box[0] * SCALE_RATIO + 0.25 * CAP_WIDTH)
                        box_r = min(CAP_WIDTH, box[2] * SCALE_RATIO + 0.25 * CAP_WIDTH)
                        visualisation = cv2.rectangle(
                            visualisation, 
                            (int(box_l), int(box_top)), 
                            (int(box_r), int(box_bottom)),
                            (0,255,0), 
                            2
                        ) 

                        visualisation = cv2.putText(
                            visualisation,
                            f"{classes[i]}, {scores[i]:.2f}",
                            (int(box_l),int(box_top*0.95)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            1
                        )
                det_time =  time.time() - tic
                if(cap.frame_num%10==0):
                    print(det_time)
                fps = 1/det_time
                fpss.append((fps, detection))
                visualisation = cv2.putText(visualisation, "{:.2f}".format(fps),(12,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                #cv2.imshow("Camera", visualisation)
                if(videostarttime is None):
                    videostarttime=time.time()
                #if(not writing_started):
                #    out.write(visualisation, 9)
                #    writing_started = True
                #out.frame = visualisation
                # keyCode = cv2.waitKey(1) & 0xFF
                # if keyCode == 27:
                #     break
                #print("whole time", time.time() - tic)
                tic = time.time()
            time.sleep(0.5)
            #out.writing = False
            duration = time.time() - starttime
            print("duration", duration)
            print("videoduration", time.time() - videostarttime)
            
            no_det_fpss = [f[0] for f in fpss if not f[1]][2:]
            det_fpss = [f[0] for f in fpss if f[1]][2:]
            all_fpss = [f[0] for f in fpss]
            
            avg_fps_det = statistics.mean(det_fpss)
            std_fps_det = statistics.stdev(det_fpss)
            max_fps_det = max(det_fpss)
            min_fps_det = min(det_fpss)

            avg_fps_nodet = statistics.mean(no_det_fpss)
            std_fps_nodet = statistics.stdev(no_det_fpss)
            max_fps_nodet = max(no_det_fpss)
            min_fps_nodet = min(no_det_fpss)

            avg_fps_all = statistics.mean(all_fpss)
            std_fps_all = statistics.stdev(all_fpss)
            max_fps_all = max(all_fpss)
            min_fps_all = min(all_fpss)

            with open("results_{}.txt".format(datetime.datetime.now()),'w+') as f:
                f.write(f"filename: {filename}\n")
                f.write(f"duration: {duration}\n")
                f.write(f"\nframes with detected signs:\n")
                f.write(f"avg fps: {avg_fps_det}\n")
                f.write(f"std fps: {std_fps_det}\n")
                f.write(f"max fps: {max_fps_det}\n")
                f.write(f"min fps: {min_fps_det}\n")

                f.write(f"\nframes without detected signs:\n")
                f.write(f"avg fps: {avg_fps_nodet}\n")
                f.write(f"std fps: {std_fps_nodet}\n")
                f.write(f"max fps: {max_fps_nodet}\n")
                f.write(f"min fps: {min_fps_nodet}\n")

                f.write(f"\nframes total:\n")
                f.write(f"avg fps: {avg_fps_all}\n")
                f.write(f"std fps: {std_fps_all}\n")
                f.write(f"max fps: {max_fps_all}\n")
                f.write(f"min fps: {min_fps_all}\n")

                f.write("\ndetections:\n")
                for d in detections:
                    f.write(f"{d[0]}, {d[1]}, second {d[2]/FPS}\n")    

            cap.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            out.writing = False
            print("stopped")
            cap.release()
            cv2.destroyAllWindows()


def record_and_save(bt_server):
    '''
    method for recording and saving the video file
    '''
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, framerate=FPS+1), cv2.CAP_GSTREAMER) 	
    while not bt_server.detecting:
        _, img = cap.read()
        if(bt_server.test_required):
            bt_server.sendTestImage(img)

    name = "output{}".format(datetime.datetime.now().time())
    _out = cv2.VideoWriter('/home/pkos/workspace/praca_inzynierska/jetson_app/outputs/{}.mp4'.format(name), 
        cv2.VideoWriter_fourcc(*'MP4V'), FPS, (CAP_WIDTH,CAP_HEIGHT))
    out = CamVideoWriter(_out)

    fpss =0
    i=0
    start = time.time()
    frames = []
    try:
        while bt_server.detecting:
            i=i+1
            tic = time.time()
            ret, frame = cap.read()
            #print("read_fps:", 1/(time.time()-tic))
            if ret:
                #frame = cv2.flip(frame,0)
                frames.append(frame)
                if(len(frames)==30):
                    out.write(frames)
                    frames=[]
                #cv2.imshow('frame',frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                if(bt_server.test_required):
                    bt_server.sendTestImage(frame)
            else:
                break
            try:
                bt_server.socket.getpeername()
            except:
                bt_server.detecting = False

                #bt_server.startAccepting()
            #time.sleep(max(0,(1/FPS) - (time.time() - tic)))
            fps = 1/(time.time() - tic)
            fpss=fpss + fps#.append(fps)
            if(i%(FPS*5)==0):
                print(fpss/i)
                print(str(out.next_id),"/", str(out.curr_id))
                i=0
                fpss=0
        duration = time.time() - start
        while(out.curr_id > out.next_id):
            print(str(out.next_id),"/",str(out.curr_id))
            time.sleep(2)
       # out = cv2.VideoWriter('/home/pkos/workspace/praca_inzynierska/jetson_app/outputs/{}.mp4'.format(name), 
       #     cv2.VideoWriter_fourcc(*'MP4V'), FPS, (CAP_WIDTH,CAP_HEIGHT))
        #[out.write(f) for f in frames]
        write_duration =time.time()-start-duration
        with open("/home/pkos/workspace/praca_inzynierska/jetson_app/outputs/{}.txt".format(name), 'w+') as f:
            f.write(str(duration))
            f.write("\n")
            f.write(str(write_duration))
        print("duration", duration)
        print("write duration", time.time() - start - duration)
    except KeyboardInterrupt:
        print("duration", time.time() - start)
        # cap.release()i
        # out.release()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    bt_server = BTServer()
    bt_server.startAdvertising()

    cfg = get_cfg()    # obtain detectron2's default config
    cfg.merge_from_file("config.yml")
    cfg.MODEL.WEIGHTS = os.path.join("./model_final.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.35
    cfg.INPUT.MIN_SIZE_TEST = CROPPED_HEIGHT
    cfg.INPUT.MAX_SIZE_TEST = CROPPED_WIDTH
    predictor = MTSDPredictor(cfg)
    print("loaded predictor")
    #run_video_inference(predictor, "./outputs/output11_13.mp4")#predictor)
    run_camera_inference(bt_server, predictor)
    #record_and_save(bt_server)
