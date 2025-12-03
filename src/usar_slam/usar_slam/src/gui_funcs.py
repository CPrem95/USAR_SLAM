import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt, QTimer


def tag_gui(index_pipe, deploy_pipe):
    """Create a GUI to display the status of the tags."""
    class ColorIndicator(QWidget):
        def __init__(self, color):
            super().__init__()
            self.layout = QVBoxLayout()
            self.label = QLabel('_', self)
            self.label.setStyleSheet("font-size: 18px; font-weight: bold; color: black;")
            self.label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label)
            self.setAutoFillBackground(True)
            self.set_color(color)

        def set_color(self, color):
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor(color))
            self.setPalette(palette)
            
        def set_text(self, text):
            """Set the label text."""
            self.label.setText(text)

    class TagWindow(QWidget):
        def __init__(self):
            super().__init__()
            
            # Adjust window size
            self.setFixedSize(300, 500)
            self.setWindowTitle('GUI')
            self.initUI()
            
            # Setup a QTimer to check for messages from the main process
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.check_pipe)
            self.timer.start(100)  # Poll every 100ms
            
            self.index_pipe = index_pipe
            
        def initUI(self):
            layout = QVBoxLayout()
            self.adjustSize()
            # Create 3 color indicators
            self.indicator = [ColorIndicator('blue'), ColorIndicator('blue'), ColorIndicator('blue'), ColorIndicator('blue')]

            layout.addWidget(self.indicator[0])
            layout.addWidget(self.indicator[1])
            layout.addWidget(self.indicator[2])
            layout.addWidget(self.indicator[3])
            
            self.indicator[0].set_text("Tag_1 NOT deployed")
            self.indicator[1].set_text("Tag_2 NOT deployed")
            self.indicator[2].set_text("Tag_3 NOT deployed")
            self.indicator[3].set_text("Tag_4 NOT deployed")

            # Create buttons to change color
            btn1 = QPushButton("Deploy Tag_1")
            btn2 = QPushButton("Deploy Tag_2")
            btn3 = QPushButton("Deploy Tag_3")
            btn4 = QPushButton("Deploy Tag_4")

            # btn1.clicked.connect(lambda: self.set_color(0, 'red'))
            # btn2.clicked.connect(lambda: self.set_color(1, 'red'))
            # btn3.clicked.connect(lambda: self.set_color(2, 'red'))
            btn1.clicked.connect(lambda: self.deploy_callback(0))
            btn2.clicked.connect(lambda: self.deploy_callback(1))
            btn3.clicked.connect(lambda: self.deploy_callback(2))
            btn4.clicked.connect(lambda: self.deploy_callback(3))

            layout.addWidget(btn1)
            layout.addWidget(btn2)
            layout.addWidget(btn3)
            layout.addWidget(btn4)

            self.setLayout(layout)
            
        def check_pipe(self):
            """Check if there's a new command from the main process."""
            if self.index_pipe.poll():
                index = self.index_pipe.recv()
                if index == -1:
                    self.set_color(0, 'red')
                    self.indicator[0].set_text("Deploy Tag_1 !")
                elif index == -2:
                    self.set_color(1, 'red')
                    self.indicator[1].set_text("Deploy Tag_2 !")
                elif index == -3:
                    self.set_color(2, 'red')
                    self.indicator[2].set_text("Deploy Tag_3 !")
                elif index == -4:
                    self.set_color(3, 'red')
                    self.indicator[3].set_text("Deploy Tag_4 !")

                elif index == 10:
                    self.set_color(0, 'green')
                    self.indicator[0].set_text("Tag_1 initialized !!")
                elif index == 11:
                    self.set_color(1, 'green')
                    self.indicator[1].set_text("Tag_2 initialized !!")
                elif index == 12:
                    self.set_color(2, 'green')
                    self.indicator[2].set_text("Tag_3 initialized !!")
                elif index == 13:
                    self.set_color(3, 'green')
                    self.indicator[3].set_text("Tag_4 initialized !!")
                    
        def deploy_callback(self, index):
            """Callback function for the deploy button."""
            if index == 0:
                self.set_color(0, 'yellow')
                self.set_text(0, "deployed, WAIT!!")
                deploy_pipe.send(0)
            elif index == 1:
                self.set_color(1, 'yellow')
                self.indicator[1].set_text("deployed, WAIT!!")
                deploy_pipe.send(1)
            elif index == 2:    
                self.set_color(2, 'yellow')
                self.indicator[2].set_text("deployed, WAITT!!")
                deploy_pipe.send(2)
            elif index == 3:
                self.set_color(3, 'yellow')
                self.indicator[3].set_text("deployed, WAITT!!")
                deploy_pipe.send(3)
                
        def set_color(self, index, color):
            self.indicator[index].set_color(color)
        
        def set_text(self, index, text):
            self.indicator[index].set_text(text) 
            
    app = QApplication(sys.argv)
    window = TagWindow()
    window.show()
    sys.exit(app.exec_())
