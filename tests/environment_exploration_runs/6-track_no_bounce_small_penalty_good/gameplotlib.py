import numpy as np
import copy
import os
import time
import pygame
from pygame.locals import *
from abc import ABC, abstractmethod
from typing import Callable, Iterable

# TODO: add axes
# TODO: finish help()

class Drawable(ABC):
    """
    abstract base class for drawable objects
    """

    def __init__(self, color: tuple[int, int, int] = (255, 255, 255),
                 follow: bool = True) -> None:

        self.color = color
        self.follow = follow # whether or not the camera should 'follow' the object by taking it into account for limit calculations


class Text(Drawable):

    def __init__(self, text: str, x: float = 0, y: float = 0,
                 color: tuple[int, int, int] = (255, 255, 255),
                 font_size: int = 15, alignment: str = "center", follow: bool = True) -> None:

        super().__init__(color, follow)

        self.text = text
        self.x = x
        self.y = y
        self.font_size = font_size
        self.alignment = alignment



class Plot(Drawable):

    def __init__(self, x: Iterable, y: Iterable,
                 color: tuple[int, int, int] = (255, 255, 255),
                 line_width: int = 1, follow: bool = True) -> None:

        super().__init__(color, follow)

        self.x = x
        self.y = y
        self.line_width = line_width


class Scatter(Drawable):

    def __init__(self, x: Iterable, y: Iterable,
                 color: tuple[int, int, int] = (255, 255, 255),
                 point_size: int = 1, follow: bool = True) -> None:

        super().__init__(color, follow)

        self.x = x
        self.y = y
        self.point_size = point_size



class Figure:

    def __init__(self, fig_size: tuple=(640, 400), bg_color: tuple=(28,28,28),
                 fps: int=60, equal_axes: bool=True, dynamic: bool=True,
                 name: str="Figure", buffer: float=0.05) -> None:

        self._running = True
        self._display_surf = None
        self.fig_size = self.figure_width, self.figure_height = fig_size[0], fig_size[1]
        self.bg_color = bg_color
        self.user_variables = {}
        self.plots = [] # will contain the user variables specific for each plot along with the options
        self.on_loop_f = lambda s, u, p: None # dummy loop function
        self.fps = fps
        self.x_max =  self.figure_width/2
        self.x_min = -self.figure_width/2
        self.y_max =  self.figure_height/2
        self.y_min = -self.figure_height/2
        self.equal_axes = equal_axes
        self.dynamic = dynamic
        self.buffer = buffer # percentage of the figure span to be used to size the buffer for each limit
        self.name = name
        self.plot_count = 0
        self.scatter_count = 0
        self.text_count = 0
        self.recording_buffer = []
        self.recording = False


    def plot(self, x: Iterable, y: Iterable, name: str = None, color: tuple=(255, 255, 255), line_width: int = 1,
             follow: bool = True) -> None:
        """
        simplified analog to matplotlib.pyplot.plot
        """

        if isinstance(x, list):
            x = np.array(x)

        if isinstance(y, list):
            y = np.array(y)

        # validate inputs 
        if len(x) != len(y):
            raise ValueError("iterables must have the same size")

        if len(x) < 2:
            raise ValueError("length of iterables must be > 1")


        # register plot
        if name is None:
            name = f"_plot_{self.plot_count}"
            self.plot_count += 1

        if name in self.user_variables:
            raise ValueError("repeated variable names are prohibited")

        self.user_variables[name] = Plot(x, y, color, line_width, follow)


    def text(self, text: str, x: float, y: float, name: str = None, color: tuple=(255, 255, 255),
             font_size: int = 15, alignment: str = "center", follow: bool = True) -> None:

        # register text
        if name is None:
            name = f"_text_{self.text_count}"
            self.text_count += 1

        if name in self.user_variables:
            raise ValueError("repeated variable names are prohibited")

        self.user_variables[name] = Text(text, x, y, color, font_size, alignment, follow)

    def scatter(self, x: Iterable, y: Iterable, name: str = None, color: tuple=(255, 255, 255), point_size: float = 1,
                follow: bool = True) -> None:
        """
        simplified analog to matplotlib.pyplot.scatter
        """

        if isinstance(x, list):
            x = np.array(x)

        if isinstance(y, list):
            y = np.array(y)

        # validate inputs 
        if len(x) != len(y):
            raise ValueError("iterables must have the same size")

        if len(x) < 2:
            raise ValueError("length of iterables must be > 1")


        # register plot
        if name is None:
            name = f"_scatter_{self.scatter_count}"
            self.scatter_count += 1

        if name in self.user_variables:
            raise ValueError("repeated variable names are prohibited")

        self.user_variables[name] = Scatter(x, y, color, point_size, follow)


    def add_variable(self, var, var_name: str) -> None:

        if var_name in self.user_variables:
            raise ValueError("repeated variable names are prohibited")

        self.user_variables[var_name] = var

    def delete_variable(self, var_name: str) -> None:

        del self.user_variables[var_name]

    def clear_variables(self) -> None:

        self.user_variables = {}
        self.plot_count = 0
        self.scatter_count = 0
        self.text_count = 0

    def re_render(self) -> None:
        self._on_render()

    def set_on_loop(self, on_loop_f: Callable) -> None:
        """
        :param _on_loop_f: a function with signature _on_loop_f(user_variables: dict, pressed_keys: list) -> None where user_variables can be used to access
                          all previously defined user variables. pressed_keys is a list which can be used with the constants defined in pygame.locals 
        """
        self.on_loop_f = on_loop_f

    def set_hang(self) -> None:
        """
        clears the on_loop_f function to just hang until window is closed
        """

        self.on_loop_f = lambda s, u, p: None # dummy loop function

    def start_recording(self, filename: str) -> None:

        if (filename[-4:] != ".mp4") and (filename[-4:] != ".mov"): 
            raise ValueError("filename must end in either '.mp4' or '.mov'")

        self.recording = True
        self.recording_filename = filename
        time.sleep(0.5)

    def stop_recording(self) -> None:

        if not self.recording:
            raise RuntimeError("can't stop a recording that never started")

        temp_dir_name = f"__{os.getpid()}_temp__"
        os.makedirs(temp_dir_name, exist_ok=False)

        for i, frame in enumerate(self.recording_buffer):
            pygame.image.save(frame, temp_dir_name + f"/frame_{i}.png")

        frame_name = temp_dir_name + "/frame_%d.png"
        os.system(f"ffmpeg -i {frame_name} -framerate {self.fps} {self.recording_filename}")
        os.system(f"rm -r {temp_dir_name}")

        self.recording = False
        self.recording_filename = None
        self.recording_buffer = [] # <- clear recording buffer


    def _do_plots(self) -> None:
        """
        internal function to actually render the plots
        """

        # handle figure limits
        if self.dynamic:

            self._update_limits()

        # plot
        for name, variable in self.user_variables.items():

            if isinstance(variable, Plot):
                plot = variable
            else:
                continue

            x, y = self._transform_coordinates(plot.x, plot.y)

            reference = min(self.figure_width, self.figure_height)
            line_width = max(1/640 * reference * plot.line_width, 1)

            points = list(zip(x, y))
            pygame.draw.lines(self._display_surf, plot.color, False, points, int(line_width))

    def _do_scatters(self) -> None:
        """
        internal function to actually render the plots
        """

        # handle figure limits
        if self.dynamic:

            self._update_limits()

        # plot
        for name, variable in self.user_variables.items():

            if isinstance(variable, Scatter):
                scatter = variable
            else:
                continue

            x, y = self._transform_coordinates(scatter.x, scatter.y)

            
            for point in zip(x, y):
                
                reference = min(self.figure_width, self.figure_height)
                radius = max(1/640 * reference * scatter.point_size, 1)
                pygame.draw.circle(self._display_surf, scatter.color, point, radius)

    
    def _do_texts(self) -> None:
        """
        internal function to render text
        """

        # handle figure limits
        if self.dynamic:

            self._update_limits()


        # do texts
        for name, variable in self.user_variables.items():

            if isinstance(variable, Text):
                text = variable
            else:
                continue

            x, y = self._transform_coordinates(text.x, text.y)

            font = pygame.font.Font(None, text.font_size) # <- 'None' selects default font

            text_render = font.render(text.text, True, text.color) # <- 'True' selects anti-aliasing

            alignment = self._align_text(text_render, x, y, text.alignment)

            self._display_surf.blit(text_render, alignment) 
            

    def _transform_coordinates(self, x, y) -> tuple:
        """
        transform from intuitive coordinates to pixel coordinates
        """

        # get copy of data to leave originals unchanged
        x = copy.deepcopy(x)
        y = copy.deepcopy(y)

        # center (the mins are unchanged)
        x = x - self.x_bmid
        y = y - self.y_bmid

        # scale
        x = x * self.x_scalar
        y = y * self.y_scalar

        x = x + self.figure_xmid
        y = y + self.figure_ymid

        # invert y
        y = self.figure_height - y

        return x,y


    def _align_text(self, text_render: pygame.Surface, x: int, y: int, alignment: str) -> pygame.Rect:
        """
        Aligns a text surface based on the given alignment option.
        
        Parameters:
        - text (pygame.Surface): The text surface to align.
        - x (int): The x-coordinate for alignment.
        - y (int): The y-coordinate for alignment.
        - alignment (str): The alignment option (e.g., "center", "topleft", etc.).
        
        Returns:
        - pygame.Rect: The aligned rectangle.
        """
        options = {
            "center": {"center": (x, y)},
            "topleft": {"topleft": (x, y)},
            "topright": {"topright": (x, y)},
            "bottomleft": {"bottomleft": (x, y)},
            "bottomright": {"bottomright": (x, y)},
            "midtop": {"midtop": (x, y)},
            "midbottom": {"midbottom": (x, y)},
            "midleft": {"midleft": (x, y)},
            "midright": {"midright": (x, y)},
            "centerx": {"centerx": x},
            "centery": {"centery": y},
        }

        if alignment not in options:
            raise ValueError(f"Invalid alignment: {alignment}")

        return text_render.get_rect(**options[alignment])


    def _update_limits(self) -> None:
        """
        update the limits of the figure
        """

        # save old just in case
        x_max_old = self.x_max
        x_min_old = self.x_min
        y_max_old = self.y_max
        y_min_old = self.y_min

        # reset
        self.x_max = -np.inf
        self.x_min =  np.inf
        self.y_max = -np.inf
        self.y_min =  np.inf

        # unbuffered
        for name, variable in self.user_variables.items():

            if not isinstance(variable, Drawable): # <- only work consider drawables
                continue

            if not variable.follow: # <- ignore it in calculations if follow flag is not set
                continue

            x = variable.x
            y = variable.y

            if isinstance(variable, Text):

                self._update_unbuffered_limits(x, y)

            else:

                for i in range(len(x)):
                    self._update_unbuffered_limits(x[i], y[i])

        # cancel recalculation if it was incomplete
        if np.isnan(self.x_min) or np.isnan(self.x_max) or np.isnan(self.y_min) or np.isnan(self.y_max) or \
           np.isinf(self.x_min) or np.isinf(self.x_max) or np.isinf(self.y_min) or np.isinf(self.y_max):

            self.x_min = x_min_old
            self.x_max = x_max_old
            self.y_min = y_min_old
            self.y_max = y_max_old

        # buffered
        self._update_buffered_limits()


    def _update_buffered_limits(self) -> None:

        self.x_span = self.x_max - self.x_min
        self.y_span = self.y_max - self.y_min

        self.x_bmin = self.x_min - self.buffer * self.x_span
        self.x_bmax = self.x_max + self.buffer * self.x_span

        self.y_bmin = self.y_min - self.buffer * self.y_span
        self.y_bmax = self.y_max + self.buffer * self.y_span
        
        self.x_bspan = self.x_bmax - self.x_bmin
        self.y_bspan = self.y_bmax - self.y_bmin

        self.x_bmid = (self.x_bmax + self.x_bmin)/2
        self.y_bmid = (self.y_bmax + self.y_bmin)/2

        self.figure_xmid = self.figure_width/2
        self.figure_ymid = self.figure_height/2

        if self.equal_axes:
            self.x_bspan = max(self.x_bspan, self.y_bspan)
            self.y_bspan = self.x_bspan

            self.x_bmin = self.x_bmid - (self.x_bspan/2)
            self.x_bmax = self.x_bmid + (self.x_bspan/2)

            self.y_bmin = self.y_bmid - (self.y_bspan/2)
            self.y_bmax = self.y_bmid + (self.y_bspan/2)

        self.x_scalar = self.figure_width/self.x_bspan
        self.y_scalar = self.figure_height/self.y_bspan

        if self.equal_axes:
            self.x_scalar = min(self.x_scalar, self.y_scalar)
            self.y_scalar = self.x_scalar


    def _update_unbuffered_limits(self, x: float, y: float) -> None:

        if x < self.x_min:
            self.x_min = x

        elif x > self.x_max:
            self.x_max = x

        if y < self.y_min:
            self.y_min = y

        elif y > self.y_max:
            self.y_max = y

 
    def _on_init(self) -> None:
        
        # window setup
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.fig_size, pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self._display_surf.fill(self.bg_color)

        self._clock = pygame.time.Clock()

        # ready to go
        self._running = True

 
    def _on_event(self, event) -> None:

        if event.type == pygame.QUIT:
            self._running = False

        elif event.type == pygame.VIDEORESIZE:

            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            self.fig_size = self.figure_width, self.figure_height = event.w, event.h

        pygame.event.clear()


    def _on_loop(self) -> None:

        pressed_keys = pygame.key.get_pressed()

        if self.on_loop_f is not None:
            self.on_loop_f(self, self.user_variables, pressed_keys)


    def _on_render(self) -> None:

        if self.recording:
            self.recording_buffer.append(self._display_surf.copy())

        
        self._display_surf.fill(self.bg_color)
        self._do_plots()
        self._do_scatters()
        self._do_texts()
        pygame.display.flip()


    def _on_cleanup(self) -> None:

        pygame.quit()

 
    def show(self) -> None:

        if self._on_init() == False:
            self._running = False

        pygame.display.set_caption(self.name)
        self._update_limits()
 
        while( self._running ):

            for event in pygame.event.get():
                self._on_event(event)

            self._on_loop()
            self._on_render()

            self._clock.tick(self.fps)

        self._on_cleanup()

    def save(self, filename: str) -> None:

        if (filename[-4:] != ".png") and (filename[-4:] != ".jpg"): 
            raise ValueError("filename must end in either '.png' or '.jpg'")

        pygame.image.save(self._display_surf, filename)


def help() -> None:

    print("THIS FUNCTION IS STILL UNDER CONSTRUCTION")

    print("SOME HELPFUL HINTS: ")
    print("color values are RGB tuples")
    print()
    print("possible alignment values are:")
    print("  'center'")
    print("  'topleft'")
    print("  'topright'")
    print("  'bottomleft'")
    print("  'bottomright'")
    print("  'midtop'")
    print("  'midbottom'")
    print("  'midleft'")
    print("  'midright'")
    print("  'centerx'")
    print("  'centery'")


"""
EXAMPLE

import gameplotlib as gplt
import numpy as np
import time

fig = gplt.Figure(name="the name of the figure", buffer=0.1)

theta = np.linspace(0, 2*np.pi)
x = np.cos(theta)
y = np.sin(theta)

fig.scatter(x+1, y+1, "moving sc", point_size=2)
fig.plot(x, y, line_width=3)
fig.text("this is some movable text", 6, 0, "moving", font_size=30)

def on_loop_f(figure, user_variables, pressed_keys):

    increment = 0.1

    if pressed_keys[gplt.K_UP]:

        text = user_variables["moving"]
        text.y += increment

    if pressed_keys[gplt.K_DOWN]:

        text = user_variables["moving"]
        text.y -= increment

    if pressed_keys[gplt.K_LEFT]:

        text = user_variables["moving"]
        text.x -= increment

    if pressed_keys[gplt.K_RIGHT]:

        text = user_variables["moving"]
        text.x += increment

    if pressed_keys[gplt.K_r]:

        if not figure.recording:
            figure.start_recording(f"{int(time.time())}.mp4")
        else:
            figure.stop_recording()

    if pressed_keys[gplt.K_s]:

        print(user_variables)
        exit()

fig.set_on_loop(on_loop_f)

fig.show()

"""

