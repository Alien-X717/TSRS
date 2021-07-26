
import threading,pyttsx3
import cv2 as cv
import numpy as np
from PIL import Image

from keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets
model = load_model('classifier.h5')
RUNNING = True
STOPPED = False

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(460, 279)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.start_btn = QtWidgets.QPushButton(self.centralwidget)
        self.start_btn.setGeometry(QtCore.QRect(50, 130, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.start_btn.setFont(font)
        self.start_btn.setAutoFillBackground(False)
        self.start_btn.setObjectName("start_btn")
        self.stop_btn = QtWidgets.QPushButton(self.centralwidget)
        self.stop_btn.setGeometry(QtCore.QRect(280, 130, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.stop_btn.setFont(font)
        self.stop_btn.setObjectName("stop_btn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 30, 391, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 460, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Road Sign recognition"))
        self.start_btn.setText(_translate("MainWindow", "Start"))
        self.stop_btn.setText(_translate("MainWindow", "Stop"))
        self.label.setText(_translate("MainWindow", "Road Sign Recognition System"))

class classifier():
    
    def __init__(self,):
        self.classes = { 0:'No Cars',
                1:'Turn Right Ahead',
                2:'Turn Left Ahead',
                3:'Go Straight or Right',
                4:'Go Straight or Left',
                #5:'No Honking',
                #6:'Overtaking from Left Prohibited',
                7:'Cars Allowed',
                8:'Stop',
                9:'No Entry',
                10:'General Caution',
                11:'Narrow Road On Right Ahead',
                12:'Compulsory Ahead',
            }
        pass

    def classify(self,image):
        
        try:
            
            image = np.array(image)
            data=[image]
            img = np.array(data)
            pred = model.predict_classes(img)
            prob=round(np.amax(model.predict(img))*100,2)
            sign=self.classes.get(pred[0],None)
            print(str(sign)+":"+str(prob)+"%")
            
            if pred != None and sign != None:
                #cv.putText(imgContour,str(sign),(pt1,pt2),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                return sign,prob
        except Exception:
            pass    
            
        return None,None

class Alert():
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        FEMALE_VOICE = voices[1].id
        self.engine.setProperty('voice',FEMALE_VOICE)
        self.ongoing=False
        self.text=""
    
    def say(self):
        self.ongoing=True        
        self.engine.say(self.text)
        self.engine.runAndWait()
        self.ongoing=False

    def speak(self,txt):
        if self.ongoing == False:
            self.text=txt
            aud=threading.Thread(target=self.say)
            aud.start()
            


class Vision():
    def __init__(self):
        self.feed = cv.VideoCapture(0)
        self.feed.set(3,480)
        self.feed.set(4,270)
        
    
    def pre_process(self):
        #threshold1 = cv.getTrackbarPos('threshold1','parameters')
        #threshold2 = cv.getTrackbarPos('threshold2','parameters')
        blur=cv.GaussianBlur(self.frame,(5,5),1)
        canyimg = cv.Canny(blur,255,255)
        dilimg=cv.dilate(canyimg,(1,1),iterations=3)
        eroded = cv.erode(dilimg,(1,1),iterations=1)
        return eroded
        
        #return cv.erode(cv.dilate(cv.Canny(cv.GaussianBlur(self.frame,(5,5),1),255,255),(1,1),iterations=3),(1,1),iterations=1)
       
    
    def crop(self,x,y,w,h,img):
    
        pts1=np.float32([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
        pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
        mat=cv.getPerspectiveTransform(pts1,pts2)
        res=cv.warpPerspective(img,mat,(w,h))
        image=cv.resize(res,(30,30),interpolation=cv.INTER_AREA)
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        return image
    
    def detect_object(self, image):
        found = False
        objs = []
        contours, heirarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for contr in contours:
            area = cv.contourArea(contr)

            #min_area = cv.getTrackbarPos('Min_object_area', 'parameters')
            min_area=4000
            max_area=10000
            if area>min_area and area<max_area:
                peri = cv.arcLength(contr, True)

                aprox = cv.approxPolyDP(contr, 0.02*peri, True)
                sides = len(aprox)

                if sides > 2 and sides < 5:
                    #cv.drawContours(imgContour,contr,-1,(255,0,255),2)

                    #for dim in aprox:
                    #    for x,y in dim:
                    #        cv.circle(imgContour,(x,y),10,(0,0,255),1)

                    x, y, w, h = cv.boundingRect(aprox)
                    if w > h:
                        h = w
                    else:
                        w = h
                    cv.rectangle(self.imgContour, (x-10, y-10),
                                (x+w+10, y+h+10), (255, 0, 0), 2)
                    cropped = self.crop(x-10, y-10, w+10, h+10, img=self.frame)
                    found = True
                    objs.append({'data': cropped, 'pt1': x+w, 'pt2': y+h})
                    #if pred != None:1
                    #    cv.putText(imgContour,str(pred),(x+w+10,y+10),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
                    #cv.putText(imgContour,'area: '+str(int(area)),(x+w+10,y+25),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

        try:
            #thrsh1=cv.getTrackbarPos('circle_threshold1','parameters')
            #thrsh2=cv.getTrackbarPos('circle_threshold2','parameters')

            #min_rad = cv.getTrackbarPos('min_radius', 'parameters')
            min_rad=30
            circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 50,
                                    param1=40, param2=40, minRadius=min_rad, maxRadius=150)
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:

               
                #cv.circle(imgContour,(i[0],i[1]),i[2],(255,0,0),2)
               
                #cv.circle(imgContour,(i[0],i[1]),2,(255,0,0),3)

                cv.rectangle(self.imgContour, (i[0]-i[2], i[1]-i[2]),
                            (i[0]+i[2], i[1]+i[2]), (255, 0, 0), 2)
                cropped = self.crop(x=i[0]-i[2], y=i[1]-i[2],
                            w=(2*i[2])+10, h=(2*i[2])+10, img=self.frame)
                found = True
                objs.append({'data': cropped,
                            'pt1': i[0]+i[2],
                            'pt2': i[1]})
                #cv.putText(imgContour,'radius: '+str(int(i[2])),(i[0],i[1]+i[2]),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,250),1)
                #if pred != None:
                #    cv.putText(imgContour,str(pred),(i[0],i[1]+i[2]),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,250),1)
        except Exception as e:
            #print(e)
            pass

        if found==True:
            return objs
        else:
            return None
        
    def capture(self):
        
        success,self.frame = self.feed.read()
        self.imgContour=self.frame
        objs=None
        if success:
            processed_img=self.pre_process()
            objs=self.detect_object(processed_img)
        return objs
    
    def annotate(self,pt1,pt2,sign):
        cv.putText(self.imgContour,str(sign),(pt1,pt2),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

    def getImg(self):
        return cv.resize(self.imgContour,None,fx=1.5,fy=1.5,interpolation=cv.INTER_AREA)
        
        



class RoadSignRecognition(Ui_MainWindow):
    def __init__(self,*args, **kwargs):
        super(Ui_MainWindow, self).__init__(*args, **kwargs)
        self.status = STOPPED
        self.model = classifier()
        self.vision = Vision()
        self.speaker = Alert()
        
        

    def pre_start(self):
        
        self.stop_btn.setDisabled(True)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def start(self):
        self.status=RUNNING
        self.start_btn.setDisabled(True)
        self.stop_btn.setDisabled(False)
        print("start")
        while(self.status==RUNNING):
            objs = self.vision.capture()
            if objs != None and len(objs)!=0:
                for obj in objs:
                    sign,prob=self.model.classify(image=obj['data'])
                    if sign and prob:
                        if prob>80 and sign!=None:
                            self.vision.annotate(sign=sign,pt1=obj['pt1'],pt2=obj['pt2'])
                            self.speaker.speak(sign)
            
            cv.imshow("Trafic Sign Detection and Recognition System ",self.vision.getImg())      
            cv.waitKey(1)
        cv.destroyAllWindows()


    def stop(self):
        print("stop")
        self.status = STOPPED
        self.stop_btn.setDisabled(True)
        self.start_btn.setDisabled(False)
        
    



        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    obj = RoadSignRecognition()
    obj.setupUi(MainWindow)
    MainWindow.show()
    obj.pre_start()
    sys.exit(app.exec_())
