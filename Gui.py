import sys
import subprocess
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from os.path import abspath, dirname
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from os.path import abspath, dirname
from Scripts import Init as s
from Scripts import Knn_Bt as k
sys.path.insert(0, abspath(dirname(abspath(__file__)) + '/..'))

class Ui_MainWindow(object):
    def fonction(self):
        s.main()
    def fonction1(self):
        k.main()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setStyleSheet("QMainWindow{background-color:\n"
"\n"
"qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(245, 245, 245, 255), stop:0.40 rgba(245, 245, 245, 255), stop:0.80 rgba(250, 250, 250, 255), stop:1 rgba(250, 255, 250, 255))}\n"
"\n"
"\n"
"QLabel#label_Heading{\n"
"font: 55 25pt \"Century Schoolbook L\";\n"
"\n"
"}\n"
"\n"
"QPushButton{\n"
"    background-color:rgb(143, 188, 143);\n"
"}\n"
)  
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setObjectName("tabWidget")
        
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setObjectName("groupBox")
        

        ###############################################################################################################"ACCUEIL
        self.Accueil = QtWidgets.QWidget()
        self.Accueil.setObjectName("Accueil")
        self.gridLayout_ac = QtWidgets.QGridLayout(self.Accueil)
        self.gridLayout_ac.setObjectName("gridLayout_ac")
        #self.gridLayout_ac.setContentsMargins(20, 0, 20, 0)
        
        self.Accueil.setStyleSheet("QWidget#Accueil{background-color:\n"
        "\n"
        "qlineargradient(spread:pad, x1:0.1, y1:0.4, x2:0.7, y2:0.35, stop:0 rgba(152, 251, 152,1.0), stop:0.60 rgba(240, 255, 240,1.0), stop:0.78 rgba(240, 255, 255,1.0), stop:1 rgba(240, 255, 255,1.0))}\n"
        "\n")

        self.icon1 = QIcon("Icones/Accueil.png")

        self.appli = QtWidgets.QVBoxLayout(self.Accueil)
        self.appli.setObjectName("appli")
        #self.appli.setSpacing(30)
        self.appli.setContentsMargins(0, 0, 0, 0)

        self.image_appli = QImage("Icones/1.png")
        self.image_appli_affiche = QPixmap.fromImage(self.image_appli.scaled(300,100,Qt.IgnoreAspectRatio,Qt.SmoothTransformation))
        
        self.label_appli = QLabel(self.Accueil)
        self.label_appli.setObjectName("label_12")
        self.label_appli.setPixmap(self.image_appli_affiche)
        #self.appli.addWidget(self.label_appli)

        self.pushButton1 = QtWidgets.QPushButton("Classification utilisant un RN")
        self.pushButton1.setObjectName("pushButton1")
        self.pushButton1.resize(10,10)
        #self.pushButton1.setEnabled(False)
        self.pushButton1.clicked.connect(self.fonction)

        self.pushButton2 = QtWidgets.QPushButton("Classification utilisant d'autres classifieurs")
        self.pushButton2.setObjectName("pushButton2")
        self.pushButton2.resize(10,10)
        #self.pushButton1.setEnabled(False)
        self.pushButton2.clicked.connect(self.fonction1)

        self.label_imageGl = QTextEdit()
        self.label_imageGl.setReadOnly(True) 
        self.label_imageGl.setText("<h1 style = color:blue;>Cet ensemble de données</h1><p><br>comprend des travaux menés par James Cope, Charles Mallah et."
                                   " assez basique de la reconnaissance d'objet.L'idée ici est de trouver des régions identiques d'une image"
                                   " James Orwell  Charles Mallah et James Orwell,  Kingston University London..<br><br></br><i>Pour des "
                                   "Donateur de la base de données Charles / Mallah: charles.mallah'@' kingston.ac.uk / James Cope: james.cope '@' kingston.ac.uk,.")

        self.gridLayout_ac.addWidget(self.pushButton1, 7, 3, 2, 2)
        self.gridLayout_ac.addWidget(self.pushButton2, 7, 14, 2, 2)
        self.gridLayout_ac.addWidget(self.label_appli, 1, 2, 2 ,3)
        self.gridLayout_ac.addWidget(self.label_imageGl, 3, 10, 2 ,7)
        
        self.tabWidget.addTab(self.Accueil,self.icon1, "Test")
        self.verticalLayout_5.addWidget(self.tabWidget)   
        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    # setup ui
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.setWindowTitle("One-hundred plant species leaves")
    window.setWindowIcon(QIcon('Icones/Accueil.png'))
    if "--travis" in sys.argv:
        QtCore.QTimer.singleShot(2000, app.exit)
    # run
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
	
