from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
from kivy.uix.screenmanager import ScreenManager, Screen, ScreenManagerException, RiseInTransition
from kivy.uix.popup import Popup
from vision_solver import single_image, initialize_cam, capture, VisionSudokuError
import cv2
import os
import psutil
from threading import Thread

Window.size = (500, 700)


class StartWindow(Screen):

    def __init__(self, **kwargs):
        super(StartWindow, self).__init__(**kwargs)
        self.image_solver_window = ImageSolverWindow()
        self.live_feed_window = LiveFeedWindow()
        self.initialize_cam_popup = InitializeCamPopup(start=self)

    def use_image_solver(self):

        try:
            self.manager.get_screen('image_solver')
        except ScreenManagerException:
            self.manager.add_widget(self.image_solver_window)

        self.manager.current = 'image_solver'

    def use_live_feed(self):
        self.initialize_cam_popup.open()
        try:
            self.manager.get_screen('live_feed')
        except ScreenManagerException:
            self.manager.add_widget(self.live_feed_window)

    def start_live_feed(self):
        cap, frame_rate = initialize_cam()

        self.live_feed_window.cap = cap
        self.live_feed_window.frame_rate = frame_rate

        self.manager.current = 'live_feed'


class ImageSolverWindow(Screen):

    def __init__(self, **kwargs):
        super(ImageSolverWindow, self).__init__(**kwargs)
        self.solution_window = ImageSolutionWindow()

    def on_pre_enter(self, *args):
        self.ids.image_file_chooser_drive.text = 'Choose drive'
        self.ids.image_file_chooser.path = '.'
        self.ids.image_file_chooser.selection = []
        self.ids.selected_file.text = ''
        self.ids.submit_button.disabled = True

    def update_drives(self):

        drive_list = []
        disk_partitions = psutil.disk_partitions(all=True)
        for partition in disk_partitions:
            drive_list.append(partition.device)

        self.ids.image_file_chooser_drive.values = drive_list

    def update_file_path_dir(self):

        drive = self.ids.image_file_chooser_drive.text
        if drive == 'Choose drive':
            self.ids.image_file_chooser.path = '.'
        else:
            self.ids.image_file_chooser.path = drive

    def on_select_file(self, file_name):

        try:
            self.ids.selected_file.text = file_name[0]
            self.ids.submit_button.disabled = False
        except IndexError:
            self.ids.submit_button.disabled = True

    def on_back(self):
        self.manager.current = "start_window"

    def on_submit_file(self):

        file_location = self.ids.selected_file.text

        try:
            solution, timing = single_image(file_location)
            self.solution_window.update_image(solution, timing)

        except VisionSudokuError as e:
            error_popup = SolutionErrorPopup(message=str(e))
            error_popup.open()
            return

        try:
            self.manager.get_screen('solution')
        except ScreenManagerException:
            self.manager.add_widget(self.solution_window)

        self.manager.current = 'solution'


class ImageSolutionWindow(Screen):

    def __init__(self, **kwargs):
        super(ImageSolutionWindow, self).__init__(**kwargs)

    def update_image(self, frame, timing):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.solution_image.texture = texture1

        self.ids.read_time.text = f'Reading image:  {timing["input_time"]}s'
        self.ids.extract_time.text = f'Extract puzzle:  {timing["extract_time"]}s'
        self.ids.solve_time.text = f'Solving puzzle:   {timing["solving_time"]}s'
        self.ids.total_time.text = f'Total time:        {timing["total_time"]}s'

    def on_main_menu(self):
        self.manager.current = 'start_window'

    def on_back(self):
        self.manager.current = 'image_solver'


class SolutionErrorPopup(Popup):

    def __init__(self, message=None, **kwargs):
        super(SolutionErrorPopup, self).__init__(**kwargs)
        self.message = message

    def on_open(self):
        self.ids.error_message.text = self.message

    def on_confirm(self):
        self.dismiss()


class InitializeCamPopup(Popup):

    def __init__(self, start=None, **kwargs):
        super(InitializeCamPopup, self).__init__(**kwargs)
        self.start = start

    def on_open(self):
        if self.start is not None:
            self.start.start_live_feed()
        self.dismiss()


class LiveFeedWindow(Screen):

    def __init__(self, **kwargs):
        super(LiveFeedWindow, self).__init__(**kwargs)
        self.cap = None
        self.frame_rate = None
        self.kill_signal = False
        self.thread = None

    def on_pre_enter(self, *args):
        self.ids.unfreeze_button.disabled = True
        self.kill_signal = False
        self.ids.cam_feed.source = 'resources/logo.png'
        self.thread = Thread(target=capture, args=[self.cap, self.frame_rate, self], daemon=True).start()

    @mainthread
    def update(self, frame, solution=False):
        if frame is not None:
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.cam_feed.texture = texture1

        if solution:
            self.ids.unfreeze_button.disabled = False

    def unfreeze(self):
        self.thread = Thread(target=capture, args=[self.cap, self.frame_rate, self], daemon=True).start()
        self.ids.unfreeze_button.disabled = True

    def on_back(self):
        self.kill_signal = True

        if self.thread is not None:
            self.thread.join()

        self.manager.current = 'start_window'


class WindowManager(ScreenManager):

    def __init__(self, **kwargs):
        super(WindowManager, self).__init__(**kwargs)
        self.transition = RiseInTransition()


class VisionSudokuApp(App):

    def build(self):
        self.title = 'Vision Sudoku'
        self.icon = 'resources/logo.png'
        Builder.load_file('interface.kv')
        window_manager = WindowManager()
        start_window = StartWindow()
        window_manager.add_widget(start_window)

        return window_manager


if __name__ == '__main__':
    VisionSudokuApp().run()
