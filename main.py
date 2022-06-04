from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from keras.models import load_model
from kivy.uix.label import Label
class MainApp(App):
    def build(self):
        self.model = load_model('keras_model.h5',compile=False)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        self.labels=['cardboard','glass','metal','paper','plastic','trash']
        self.capture = cv2.VideoCapture(0)

        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.label = Label(text='Hello world',pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        layout.add_widget(self.image)
        layout.add_widget(self.label)
        self.save_img_button = Button(
            text="CLICK HERE",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        self.save_img_button.bind(on_press=self.take_picture)
        layout.add_widget(self.save_img_button)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout
    def load_video(self, *args):
        ret, frame = self.capture.read()
        # Frame initialize
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture
    def take_picture(self, *args):
        img = self.image_frame
        size = (224, 224)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        image_array = np.asarray(img)
        self.data[0] = image_array
        prediction = self.model.predict(self.data)
        l = max(prediction[0])
        p = prediction[0]
        for j in range(len(p)):
            if(p[j] == l):
                self.label.text = self.labels[j]
if __name__ == '__main__':
    MainApp().run()
