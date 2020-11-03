'''
Created on 20 aug. 2020

Modified base graphics_pgl implementation to use the pyglet event loop for rendering
and smooth resizing.

@author: Frits de Nijs
'''

import pyglet as pgl

from flatland.utils.graphics_pil import PILSVG
from pyglet.gl import glEnable, glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST

# Set flags for OpenGL smooth scaling.
glEnable(GL_TEXTURE_2D)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

"""
Singleton view keeping track of the most recent image, and a refresh flag indicating
to the RailViewWindow when the texture has to be updated.
"""
class PygletView:

    __instance = None

    def __new__(cls):
        if PygletView.__instance is None:
            PygletView.__instance = object.__new__(cls)
            PygletView.__instance.__is_updated = False
            PygletView.__instance.__pil_img = None
            
        return PygletView.__instance

    def has_changed(self):
        return self.__is_updated

    def update_image(self, new_img):
        self.__pil_img = new_img
        self.__is_updated = True

    def get_img(self):
        if self.__is_updated:
            self.__is_updated = False
        return self.__pil_img

"""
Override of pyglet window object to implement custom draw operation.
"""
class RailViewWindow(pgl.window.Window):

    manager = None
    texture = None
    alive = True

    def __init__(self):
        super(RailViewWindow, self).__init__(caption='Flatland Schedule Viewer', resizable=True, visible=False)

        # Load static image as resource.
        self.manager = PygletView()

        # Minimum size to prevent errors with None textures etc.
        self.set_minimum_size(8, 8)

        # Show window (start handling events)
        self.set_visible(True)

    def update_texture(self, dt):
        if self.manager.has_changed():
            pil_img = self.manager.get_img()

            if pil_img is not None:
                bytes_image = pil_img.tobytes()
                new_texture = pgl.image.ImageData(pil_img.width, pil_img.height, 'RGBA', bytes_image, pitch=-pil_img.width * 4).get_texture()
                new_texture.width = self.width
                new_texture.height = self.height
                self.texture = new_texture

    def on_draw(self):
        self.clear()
        if self.texture is not None:
            self.texture.blit(0,0)

    def on_resize(self, width, height):
        if self.texture is not None and width is not None and height is not None:
            self.texture.width = max(1, width)
            self.texture.height = max(1, height)

        pgl.window.Window.on_resize(self, width, height)

    def on_close(self):
        self.alive = False
        pgl.window.Window.on_close(self)


class PGLGL(PILSVG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = PygletView()
        self.window_open = False  # means the window has not yet been opened.
        self.closed = False  # windows has been closed (currently, we leave the env still running)

    def open_window(self):
        print("old open_window - pyglet")
        pass

    def close_window(self):
        self.closed=True

    def show(self, block=False, from_event=False):
        pil_img = self.alpha_composite_layers()
        self.view.update_image(pil_img)
