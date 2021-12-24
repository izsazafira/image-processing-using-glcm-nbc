# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'guizsa.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import cv2 as cv
import numpy as np
import pandas as pd
import imutils
from skimage.feature import greycomatrix, greycoprops 
import pickle
from sklearn.naive_bayes import GaussianNB


class Ui_Identification(object):
    def setupUi(self, Identification):
        Identification.setObjectName("Identification")
        Identification.resize(1112, 665)
        self.centralwidget = QtWidgets.QWidget(Identification)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidgetGLCM = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidgetGLCM.setGeometry(QtCore.QRect(130, 320, 501, 161))
        self.tableWidgetGLCM.setObjectName("tableWidgetGLCM")
        self.tableWidgetGLCM.setColumnCount(4)
        self.tableWidgetGLCM.setRowCount(4)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetGLCM.setHorizontalHeaderItem(3, item)
        self.PengolahanCitra = QtWidgets.QGroupBox(self.centralwidget)
        self.PengolahanCitra.setGeometry(QtCore.QRect(180, 530, 191, 81))
        self.PengolahanCitra.setObjectName("PengolahanCitra")
        self.OpenImage = QtWidgets.QPushButton(self.PengolahanCitra)
        self.OpenImage.setGeometry(QtCore.QRect(30, 20, 121, 23))
        self.OpenImage.setObjectName("OpenImage")
        self.Reset = QtWidgets.QPushButton(self.PengolahanCitra)
        self.Reset.setGeometry(QtCore.QRect(30, 50, 121, 23))
        self.Reset.setObjectName("Reset")
        self.ImagePreprocessing = QtWidgets.QGroupBox(self.centralwidget)
        self.ImagePreprocessing.setGeometry(QtCore.QRect(400, 530, 331, 81))
        self.ImagePreprocessing.setObjectName("ImagePreprocessing")
        self.resizebutton = QtWidgets.QPushButton(self.ImagePreprocessing)
        self.resizebutton.setGeometry(QtCore.QRect(20, 20, 141, 23))
        self.resizebutton.setObjectName("resizebutton")
        self.grayscalebutton = QtWidgets.QPushButton(self.ImagePreprocessing)
        self.grayscalebutton.setGeometry(QtCore.QRect(20, 50, 141, 23))
        self.grayscalebutton.setObjectName("grayscalebutton")
        self.powerbutton = QtWidgets.QPushButton(self.ImagePreprocessing)
        self.powerbutton.setGeometry(QtCore.QRect(180, 20, 141, 23))
        self.powerbutton.setObjectName("powerbutton")
        self.Proses = QtWidgets.QGroupBox(self.centralwidget)
        self.Proses.setGeometry(QtCore.QRect(760, 530, 161, 81))
        self.Proses.setObjectName("Proses")
        self.GLCM = QtWidgets.QPushButton(self.Proses)
        self.GLCM.setGeometry(QtCore.QRect(10, 20, 131, 23))
        self.GLCM.setObjectName("GLCM")
        self.NBC = QtWidgets.QPushButton(self.Proses)
        self.NBC.setGeometry(QtCore.QRect(10, 50, 131, 23))
        self.NBC.setObjectName("NBC")
        self.groupBoxHasil = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxHasil.setGeometry(QtCore.QRect(750, 300, 261, 221))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBoxHasil.setFont(font)
        self.groupBoxHasil.setObjectName("groupBoxHasil")
        self.linehasil = QtWidgets.QLineEdit(self.groupBoxHasil)
        self.linehasil.setGeometry(QtCore.QRect(10, 30, 241, 61))
        self.linehasil.setObjectName("linehasil")
        self.linehasil_2 = QtWidgets.QLineEdit(self.groupBoxHasil)
        self.linehasil_2.setGeometry(QtCore.QRect(10, 140, 241, 61))
        self.linehasil_2.setObjectName("linehasil_2")
        self.label = QtWidgets.QLabel(self.groupBoxHasil)
        self.label.setGeometry(QtCore.QRect(10, 110, 191, 21))
        self.label.setObjectName("label")
        self.groupBoxRGB = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxRGB.setGeometry(QtCore.QRect(30, 100, 241, 181))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBoxRGB.setFont(font)
        self.groupBoxRGB.setObjectName("groupBoxRGB")
        self.RGB = QtWidgets.QLabel(self.groupBoxRGB)
        self.RGB.setGeometry(QtCore.QRect(30, 30, 181, 131))
        self.RGB.setFrameShape(QtWidgets.QFrame.Box)
        self.RGB.setText("")
        self.RGB.setObjectName("RGB")
        self.groupBoxRGB_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxRGB_2.setGeometry(QtCore.QRect(300, 100, 241, 181))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBoxRGB_2.setFont(font)
        self.groupBoxRGB_2.setObjectName("groupBoxRGB_2")
        self.resize = QtWidgets.QLabel(self.groupBoxRGB_2)
        self.resize.setGeometry(QtCore.QRect(30, 30, 181, 131))
        self.resize.setFrameShape(QtWidgets.QFrame.Box)
        self.resize.setText("")
        self.resize.setObjectName("resize")
        self.groupBoxRGB_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxRGB_3.setGeometry(QtCore.QRect(570, 100, 241, 181))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBoxRGB_3.setFont(font)
        self.groupBoxRGB_3.setObjectName("groupBoxRGB_3")
        self.grayscale = QtWidgets.QLabel(self.groupBoxRGB_3)
        self.grayscale.setGeometry(QtCore.QRect(30, 30, 181, 131))
        self.grayscale.setFrameShape(QtWidgets.QFrame.Box)
        self.grayscale.setText("")
        self.grayscale.setObjectName("grayscale")
        self.groupBoxRGB_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxRGB_4.setGeometry(QtCore.QRect(840, 100, 241, 181))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBoxRGB_4.setFont(font)
        self.groupBoxRGB_4.setObjectName("groupBoxRGB_4")
        self.powerlaw = QtWidgets.QLabel(self.groupBoxRGB_4)
        self.powerlaw.setGeometry(QtCore.QRect(30, 30, 181, 131))
        self.powerlaw.setFrameShape(QtWidgets.QFrame.Box)
        self.powerlaw.setText("")
        self.powerlaw.setObjectName("powerlaw")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 0, 1081, 81))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        Identification.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Identification)
        self.statusbar.setObjectName("statusbar")
        Identification.setStatusBar(self.statusbar)

        self.retranslateUi(Identification)
        QtCore.QMetaObject.connectSlotsByName(Identification)

    def retranslateUi(self, Identification):
        _translate = QtCore.QCoreApplication.translate
        Identification.setWindowTitle(_translate("Identification", "MainWindow"))
        item = self.tableWidgetGLCM.verticalHeaderItem(0)
        item.setText(_translate("Identification", "Energy"))
        item = self.tableWidgetGLCM.verticalHeaderItem(1)
        item.setText(_translate("Identification", "Contrast"))
        item = self.tableWidgetGLCM.verticalHeaderItem(2)
        item.setText(_translate("Identification", "Correlation"))
        item = self.tableWidgetGLCM.verticalHeaderItem(3)
        item.setText(_translate("Identification", "Homogeneity"))
        item = self.tableWidgetGLCM.horizontalHeaderItem(0)
        item.setText(_translate("Identification", "0"))
        item = self.tableWidgetGLCM.horizontalHeaderItem(1)
        item.setText(_translate("Identification", "45"))
        item = self.tableWidgetGLCM.horizontalHeaderItem(2)
        item.setText(_translate("Identification", "90"))
        item = self.tableWidgetGLCM.horizontalHeaderItem(3)
        item.setText(_translate("Identification", "135"))
        self.PengolahanCitra.setTitle(_translate("Identification", "Pengolahan Citra"))
        self.OpenImage.setText(_translate("Identification", "Open Image"))
        self.Reset.setText(_translate("Identification", "Reset"))
        self.ImagePreprocessing.setTitle(_translate("Identification", "Image Preprocessing"))
        self.resizebutton.setText(_translate("Identification", "Resize"))
        self.grayscalebutton.setText(_translate("Identification", "Grayscale"))
        self.powerbutton.setText(_translate("Identification", "Power Law Transormation"))
        self.Proses.setTitle(_translate("Identification", "Proses"))
        self.GLCM.setText(_translate("Identification", "GLCM"))
        self.NBC.setText(_translate("Identification", "NBC"))
        self.groupBoxHasil.setTitle(_translate("Identification", "Hasil Klasifikasii"))
        self.label.setText(_translate("Identification", "Rekomendasi Pengobatan"))
        self.groupBoxRGB.setTitle(_translate("Identification", "RGB Image"))
        self.groupBoxRGB_2.setTitle(_translate("Identification", "Resize Image"))
        self.groupBoxRGB_3.setTitle(_translate("Identification", "Grayscale Image"))
        self.groupBoxRGB_4.setTitle(_translate("Identification", "Power Law Transformation"))
        self.label_2.setText(_translate("Identification", "KLASIFIKASI JENIS JERAWAT MENGGUNAKAN METODE GRAY LEVEL CO-OCCURENCE MATRIX DAN NAIVE BAYES CLASSIFIER"))

        self.OpenImage.clicked.connect(self.OpenImages)
        self.Reset.clicked.connect(self.functionreset)
        self.resizebutton.clicked.connect(self.OpenResize)
        self.grayscalebutton.clicked.connect(self.konversigrayscale)
        self.powerbutton.clicked.connect(self.konversipowerlaw)
        self.GLCM.clicked.connect(self.prosesglcm)
        self.NBC.clicked.connect(self.prosesNBC)
    
    def OpenImages(self):
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)")
            if fileName:
                pixmap = QtGui.QPixmap(fileName)
                pixmap = pixmap.scaled(self.RGB.width(),self.RGB.height())
                self.RGB.setPixmap(pixmap)
                self.RGB.setAlignment(QtCore.Qt.AlignCenter)
                self.resizebutton.setEnabled(True)
                self.file = fileName
  
            self.image = cv.imread(self.file, cv.IMREAD_ANYCOLOR)
            self.processedImage = self.image.copy()
            self.resizebutton.setEnabled(True)
    
    def OpenResize(self):          
         self.processedImage = self.image.copy()
         self.previewImage = imutils.resize(self.processedImage, width=128, height=128) 
         self.Displayresize()
    
    def Displayresize(self):
        qFormat = QtGui.QImage.Format_Indexed8
        if len (self.previewImage.shape) == 3:
            if (self.previewImage.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
               qFormat = QtGui.QImage.Format_RGB888
        img1 = QtGui.QImage(self.previewImage, self.previewImage.shape[1], self.previewImage.shape[0],self.previewImage.strides[0], qFormat)
        img1 = img1.rgbSwapped()
                
        
        self.resize.setPixmap(QtGui.QPixmap.fromImage(img1))
        self.resize.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.grayscalebutton.setEnabled(True)
    
    def konversigrayscale(self):
        self.gray = cv.cvtColor(self.previewImage, cv.COLOR_BGR2GRAY)
        self.previewgray= self.gray
        
        self.displayGrayscale()
    
    def displayGrayscale(self):
        qFormat = QtGui.QImage.Format_Indexed8
        if len (self.previewgray.shape) == 3:
            if (self.previewgray.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888
        self.img1 = QtGui.QImage(self.previewgray, self.previewgray.shape[1], self.previewgray.shape[0], self.previewgray.strides[0], qFormat)
        self.img1 = self.img1.rgbSwapped()
        
        self.grayscale.setPixmap(QtGui.QPixmap.fromImage(self.img1))
        self.grayscale.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.powerbutton.setEnabled(True)
        
    def konversipowerlaw(self):
        self.power = np.array(255*(self.previewgray/255)**1.5,dtype='uint8')
        img3 = cv.hconcat([self.power])
        self.previewpower = img3
         
        self.displaypower()
    
    def displaypower(self):
        qFormat = QtGui.QImage.Format_Indexed8
        if len (self.previewpower.shape) == 3:
            if (self.previewpower.shape[2]) == 4:
                qFormat = QtGui.QImage.Format_RGBA8888
            else:
                qFormat = QtGui.QImage.Format_RGB888
        self.img1 = QtGui.QImage(self.previewpower, self.previewpower.shape[1], self.previewpower.shape[0], self.previewpower.strides[0], qFormat)
        self.img1 = self.img1.rgbSwapped()
        
        self.powerlaw.setPixmap(QtGui.QPixmap.fromImage(self.img1))
        self.powerlaw.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.GLCM.setEnabled(True)
    
    def prosesglcm(self):
        L = 256
        def getAngledGLCM(img3, alpha):
            row, col = img3.shape
            # row, col, layer = img3.shape
            glcm = np.zeros((L, L), np.uint8)
            xy = [0, 0]
            rowStart = 0
            colStart = 0
            rowEnd = row
            colEnd = col - 1
        
            if alpha == 0:
                xy = [0, 1]
            elif alpha == 45:
                xy = [-1, 1]
                rowStart = 1
                colStart = 0
            elif alpha == 90:
                xy = [-1, 0]
                rowStart = 1
                colEnd = col
            else:
                xy = [-1, -1]
                rowStart = 1
                colStart = 1
                colEnd = col
        
            for x in range(rowStart, rowEnd):
                for y in range(colStart, colEnd):
                    pixVal = img3[x, y]
                    pixValNeighbor = img3[x + xy[0], y + xy[1]]
                    glcm[pixVal, pixValNeighbor] += 1
        
            return glcm / glcm.sum()
        
        def getGLCM(img3, alpha):
            
            glcm = getAngledGLCM(img3, alpha)
        
            return glcm
        
        def getSumGLCM(glcm):
            sumX = []
            sumY = []
        
            for i in range(L):
                sumY.append(glcm[:, i].sum())
                sumX.append(glcm[i, :].sum())
        
            return [sumX, sumY]
        
        def getMean(glcm, sumX, sumY):
            # sumX, sumY = getSumGLCM(glcm)
            meanX = 0.0
            meanY = 0.0
        
            for i in range(L):
                meanX += i * sumX[i]
                meanY += i * sumY[i]
        
            return [meanX, meanY]
        
        def getVarianceXY(glcm, sumX, sumY, meanX, meanY):
            # meanX, meanY = getMean(glcm)
            # sumX, sumY = getSumGLCM(glcm)
            varX = 0.0
            varY = 0.0
        
            for i in range(L):
                varX += ((i - meanX) ** 2) * sumX[i]
                varY += ((i - meanY) ** 2) * sumY[i]
        
            return [varX, varY]
        
        def getStandardDeviation(varX, varY):
            # varX, varY = getVarianceXY(glcm)
        
            return [np.sqrt(varX), np.sqrt(varY)]
        
        def getASM(glcm):
            return np.power(glcm.flatten(), 2).sum()
        
        def getContrast(glcm):
            con = 0.0
        
            for x in range(L):
                for y in range(L):
                    con += ((x - y) ** 2) * glcm[x, y]
        
            return con
        
        def getCorrelation(glcm, meanX, meanY, sdX, sdY):
            cor = 0.0
        
            for x in range(L):
                for y in range(L):
                    cor += (x * y) * glcm[x, y]
        
            return (cor - (meanX * meanY)) / (sdX * sdY)
        
        def getIDM(glcm):
            idm = 0.0
          
            for x in range(L):
                for y in range(L):
                    idm += glcm[x, y] / (1 + ((x - y) ** 2))
        
            return idm
        
            glcm = np.zeros((L, L), np.float64)
            for i in range(4):
                glcm = getGLCM(self.previewgray, alpha)
                sumX, sumY = getSumGLCM(glcm)
                meanX, meanY = getMean(glcm, sumX, sumY)
                varX, varY = getVarianceXY(glcm, sumX, sumY, meanX, meanY)
                sdX, sdY = getStandardDeviation(varX, varY)
                mean = [meanX, meanY]
                asm = getASM(glcm)
                contrast = getContrast(glcm)
                correlation = getCorrelation(glcm, meanX, meanY, sdX, sdY)
                idm = getIDM(glcm)
                
                dict["asm_{}".format(alpha)] = asm
                dict["contrast_{}".format(alpha)] = contrast
                dict["correlation_{}".format(alpha)] = correlation
                dict["idm_{}".format(alpha)] = idm
                alpha += 45
                
                res.append(dict)
        if __name__ == '__main__':
            #data_path = self.previewgray
           
            dict = {}
            test = pd.DataFrame()
        
            res = []
            
            #img = cv.imread(data_path)
            glcm = np.zeros((L, L), np.float64)
            # label.append(label_dict[category])
            alpha = 0
            dict = {}

            for i in range(4):
                glcm = getGLCM(self.previewpower, alpha)
                sumX, sumY = getSumGLCM(glcm)
                meanX, meanY = getMean(glcm, sumX, sumY)
                varX, varY = getVarianceXY(glcm, sumX, sumY, meanX, meanY)
                sdX, sdY = getStandardDeviation(varX, varY)
                mean = [meanX, meanY]
                asm = getASM(glcm)
                contrast = getContrast(glcm)
                correlation = getCorrelation(glcm, meanX, meanY, sdX, sdY)
                idm = getIDM(glcm)

                dict['asm_{}'.format(alpha)] = asm
                dict['contrast_{}'.format(alpha)] = contrast
                dict['correlation_{}'.format(alpha)] = correlation
                dict['idm_{}'.format(alpha)] = idm
                alpha += 45

            res.append(dict)
        self.glcm = pd.DataFrame(res)
        #print(self.glcm)
        #self.feature = []
        #for res in self.glcm:
            #self.feature.append(res)
        #print(self.feature)
        self.tableWidgetGLCM.setItem(0 , 0, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,0])))
        self.tableWidgetGLCM.setItem(1 , 0, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,1])))
        self.tableWidgetGLCM.setItem(2 , 0, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,2])))
        self.tableWidgetGLCM.setItem(3 , 0, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,3])))
       
        self.tableWidgetGLCM.setItem(0 , 1, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,4])))
        self.tableWidgetGLCM.setItem(1 , 1, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,5])))
        self.tableWidgetGLCM.setItem(2 , 1, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,6])))
        self.tableWidgetGLCM.setItem(3 , 1, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,7])))
        
        self.tableWidgetGLCM.setItem(0 , 2, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,8])))
        self.tableWidgetGLCM.setItem(1 , 2, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,9])))
        self.tableWidgetGLCM.setItem(2 , 2, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,10])))
        self.tableWidgetGLCM.setItem(3 , 2, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,11])))
        
        self.tableWidgetGLCM.setItem(0 , 3, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,12])))
        self.tableWidgetGLCM.setItem(1 , 3, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,13])))
        self.tableWidgetGLCM.setItem(2 , 3, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,14])))
        self.tableWidgetGLCM.setItem(3 , 3, QtWidgets.QTableWidgetItem(str(self.glcm.iloc[0,15])))
        
        self.NBC.setEnabled(True)

    
    def prosesNBC(self):
        model = pickle.load(open('model_training data45 74.sav', 'rb'))
        data = [self.glcm.loc[0]]
        hasil = model.predict(data)
        if hasil[0] == 0:
          self.linehasil.setText("Nodul")
          self.linehasil_2.setText("Periksa Ke Dokter")
        if hasil[0] == 1:
          self.linehasil.setText("Papula")
          self.linehasil_2.setText("Periksa Ke Dokter")
        if hasil[0] == 2:
          self.linehasil.setText("Pustula")
          self.linehasil_2.setText("Asam Salisilat")
        
    def functionreset(self):
        self.RGB.clear()
        self.resize.clear()
        self.grayscale.clear()
        self.powerlaw.clear()
        self.tableWidgetGLCM.clearContents()
        self.linehasil.setText("")
        self.linehasil_2.setText("")
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Identification = QtWidgets.QMainWindow()
    ui = Ui_Identification()
    ui.setupUi(Identification)
    Identification.show()
    sys.exit(app.exec_())
