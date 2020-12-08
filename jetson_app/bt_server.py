import bluetooth, threading, time, cv2
from detection_utils import extract_box, nms, handle_prediction
from camera_parameters import *
import datetime
LBL_DATA_DELIMITER = '&&&'
DATA_START_SYMBOL = '###'
SIZE_START_SYMBOL = '$$$'
START_DET_COMMAND = 'START'
STOP_DET_COMMAND = 'STOP'
SEND_TEST_IMG_COMMAND = 'TEST CAMERA'

class BTServer:
    '''
	server is used to send data to user's phone and handles user's commands like stop/start detection
	'''

    def __init__(self):
        self.accepting = False
        self.test_required = False
        self.sending = False
        self.detecting = False

    def startAdvertising(self):
        t = threading.Thread(target=(self._advertise))
        t.daemon = True
        t.start()
        return self

    def startAccepting(self):
        if not self.accepting:
            t = threading.Thread(target=(self._accept))
            t.daemon = True
            t.start()
        return self

    def startSendingDetections(self, outputs, im):
        self.outputs = outputs
        self.im = im
        t = threading.Thread(target=(self._sender))
        t.daemon = True
        t.start()
        return self

    def startReceiving(self):
        t = threading.Thread(target=(self._receive))
        t.daemon = True
        t.start()
        return self

    def _receive(self):
        while not self.test_required:
            command = self.socket.recv(1024)
            if command.decode('utf-8') == SEND_TEST_IMG_COMMAND:
                print('required')
                self.test_required = True
            if command.decode('utf-8') == START_DET_COMMAND:
                print('started detection')
                self.detecting = True
            if command.decode('utf-8') == STOP_DET_COMMAND:
                print('paused detection')
                self.detecting = False

    def sendTestImage(self, im):
        img_bytes = cv2.imencode('.jpg', im)[1].tostring()
        data_size = len(img_bytes)
        while self.sending:
            time.sleep(0.5)

        self.socket.send(SIZE_START_SYMBOL + str(data_size))
        print('sent size')
        self.socket.sendall(DATA_START_SYMBOL.encode() + img_bytes)
        self.test_required = False
        self.startReceiving()

    def _accept(self):
        self.accepting = True
        self.socket, address = self.serverSocket.accept()
        print('Got connection with', address)
        self.accepting = False
        self._receive()

    def _advertise(self):
        name = 'bt_server'
        target_name = 'test'
        uuid = '94f39d29-7d6d-437d-973b-fba39e49d4ee'
        self.serverSocket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        port = bluetooth.PORT_ANY
        self.serverSocket.bind(('', port))
        print('Listening for connections on port: ', port)
        self.serverSocket.listen(1)
        port = self.serverSocket.getsockname()[1]
        bluetooth.advertise_service((self.serverSocket),
          'RoadeyeServer',
          service_id=uuid,
          service_classes=[
         uuid, bluetooth.SERIAL_PORT_CLASS],
          profiles=[
         bluetooth.SERIAL_PORT_PROFILE])
        self._accept()

    def _sender(self):
        tic = time.time()
        detection_time = str(datetime.datetime.now().strftime('%c'))
        boxes, classes, scores = handle_prediction(self.outputs)
        boxes, classes, scores = nms(boxes, classes, scores, 0.5)
        print('num of detections:', len(boxes))
        while self.sending:
            time.sleep(0.5)

        try:
            self.sending = True
            for i, box in enumerate(boxes):
                print(classes[i])
                box_im = extract_box(box, self.im)
                img_bytes = cv2.imencode('.jpg', box_im)[1].tostring()
                label_and_img = (DATA_START_SYMBOL + classes[i] + LBL_DATA_DELIMITER).encode() + img_bytes
                data_size = len(label_and_img) - len(DATA_START_SYMBOL)
                print(data_size)
                print(classes[i])
                self.socket.send(SIZE_START_SYMBOL + str(data_size))
                self.socket.sendall(label_and_img)

        except:
            pass

        self.sending = False
