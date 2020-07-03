'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':1, 'slow':10, 'medium':5}
        self.max_width,self.max_height = pyautogui.size()
        
        #pyautogui.FAILSAFE=False
        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]


    def move(self, x, y):
        dx=x
        dy=y
        
        current_x,current_y = pyautogui.mouseinfo.position()
        if current_x-dx < 0:
            dx=0
        elif current_x+dx>=self.max_width:
            dx=self.max_width-1

        if current_y-dy < 0:
            dy=0
        elif current_y+dy>=self.max_height:
            dy=self.max_height-1        
        pyautogui.moveRel(dx*self.precision, -1*dy*self.precision, duration=self.speed)
