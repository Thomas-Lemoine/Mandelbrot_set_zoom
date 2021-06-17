import pygame
import numpy as np
from numba import njit, prange
import time
import random
import os

#Do not change
WIDTH = 800
HEIGHT = 800
COLS = WIDTH
ROWS = HEIGHT

THRESH = 150
ZOOM_INCR = 1.5

@njit()
def change_ranges(x0, x1, y0, y1, up_bool, zoom_factor, zoom_incr, mouseX, mouseY):
    
    if up_bool:
        scale = zoom_incr
    else:
        scale = 1/zoom_incr
    zoom_factor *= scale

    len_screen_x = x1 - x0
    len_screen_y = y1 - y0
    screen_ratio_x = mouseX/WIDTH
    screen_ratio_y = mouseY/HEIGHT
    dx = screen_ratio_x*len_screen_x
    dy = screen_ratio_y*len_screen_y
    x, y = x0 + dx, y0 + dy
    
    post_len_x = len_screen_x/scale
    post_len_y = len_screen_y/scale
    post_x0 = x - dx/scale
    post_y0 = y - dy/scale

    post_x1 = post_x0 + post_len_x
    post_y1 = post_y0 + post_len_y

    return post_x0, post_x1, post_y0, post_y1, zoom_factor


@njit(fastmath=True)
def mandelbrot_val(x, y, THRESH):
    a = 0
    b = 0
    val = 0
    for iter_count in range(THRESH):
        if (a*a + b*b > 4):
            val = 255*iter_count/THRESH #+ 1 - np.log(np.log2(np.sqrt(a*a + b*b)))
            break
        temp_a = a*a - b*b + x
        b = 2*a*b + y
        a = temp_a
    return val

@njit(parallel = True, fastmath = True)
def new_pix_arr(x0, y0, x1, y1, COLS, ROWS, THRESH):
    pixel_array = np.empty((ROWS, COLS))
    x_len = x1 - x0
    y_len = y1 - y0

    for col in prange(COLS):
        for row in prange(ROWS):
            x = x0 + x_len*col/COLS
            y = y0 + y_len*row/ROWS
            pixel_array[col, row]  = mandelbrot_val(x, y, THRESH)
    return pixel_array

def curr_x_val(x0, x1, mouse_x):
    return (x1 - x0) * mouse_x/WIDTH

def curr_y_val(y0, y1, mouse_y):
    return (y1 - y0) * mouse_y/HEIGHT

def main():
    global ROWS, COLS
    global THRESH
    global ZOOM_INCR
    zoom_incr = ZOOM_INCR
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mandelbrot Set")
    print("Press c to see a list of available commands")

    x0, x1 = -2, 2
    y0, y1 = -2, 2

    zoom_factor = 1
    quit = False

    changed = True
    mouse_dragging = False
    mouse_zoom = False

    while not quit:
        quit = pygame.event.get(pygame.QUIT)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and not changed:
                mouse_x, mouse_y = event.pos
                if event.button == 4:
                    x0, x1, y0, y1, zoom_factor = change_ranges(x0, x1, y0, y1, True, zoom_factor, zoom_incr, mouse_x, mouse_y)
                if event.button == 5: #and zoom_factor > 1: 
                    x0, x1, y0, y1, zoom_factor = change_ranges(x0, x1, y0, y1, False, zoom_factor, zoom_incr, mouse_x, mouse_y)
                if event.button == 1:
                    x = curr_x_val(x0,x1,mouse_x)
                    y = curr_y_val(y0,y1,mouse_y)
                    mouse_dragging = True
                if event.button == 3:
                    first_x = x0 + curr_x_val(x0,x1,mouse_x)
                    first_y = y0 + curr_y_val(y0,y1,mouse_y)
                    #print(first_x, first_y)
                    mouse_zoom = True
                changed = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and mouse_dragging:
                    mouse_dragging = False
                    changed = True
                if event.button == 3 and mouse_zoom:
                    mouse_x, mouse_y = event.pos
                    sec_x = x0 + curr_x_val(x0,x1,mouse_x)
                    sec_y = y0 + curr_y_val(y0,y1,mouse_y)
                    #print(sec_x, sec_y)
                    x0, x1 = min(first_x,sec_x), max(first_x,sec_x)
                    y0, y1 = min(first_y,sec_y), max(first_y,sec_y)
                    changed = True
                    mouse_zoom = False

            if event.type == pygame.MOUSEMOTION and mouse_dragging:
                mouse_x, mouse_y = event.pos
                new_x = curr_x_val(x0,x1,mouse_x)
                new_y = curr_y_val(y0,y1,mouse_y)
                diff_x, diff_y = x - new_x, y - new_y
                x0, x1 = x0 + diff_x, x1 + diff_x
                y0, y1 = y0 + diff_y, y1 + diff_y
                x = new_x
                y = new_y
                changed = True

            if event.type == pygame.KEYDOWN and not changed:
                if event.key == pygame.K_UP:
                    THRESH = THRESH * 2 
                elif event.key == pygame.K_DOWN:
                    THRESH = max(THRESH/2, 20)
                if event.key == pygame.K_q:
                    zoom_incr *= 1.2
                    print(f"Increased zoom increment to {zoom_incr}")
                if event.key == pygame.K_w:
                    zoom_incr *= 0.8
                    print(f"Decreased zoom increment to {zoom_incr}")
                if event.key == pygame.K_p:
                    x,y = pygame.mouse.get_pos()
                    len_x = x1 - x0
                    len_y = y1 - y0
                    print(f"P: {x0 + len_x * x/WIDTH} + {y0 + len_y * y/HEIGHT}i.")
                if event.key == pygame.K_c:
                    print("These are the following commands:\n"
                    "Scroll with your mouse to zoom or unzoom\n"
                    "Right click and drag the mouse to zoom into a smaller rectangle within the screen\n"
                    "Left click and draw the mouse to move around the mandelbrot set"
                    "UP and DOWN keys let you change the accuracy of the drawing by changing the maximum iterations threshold\n"
                    "q and w keys let you increase and modify the zoom increment, such that you can zoom faster or slower\n"
                    "p gets the position of the mouse in the complex plane\n"
                    "s saves the current image of the pygame surface into the program's directory\n"
                    "i shows the user relevant information such as the current maximum iterations threshold, position, and resolution of the screen.\n")
                if event.key == pygame.K_i:
                    x,y = pygame.mouse.get_pos()
                    len_x = x1 - x0
                    len_y = y1 - y0
                    print(
                        f"INFORMATION:\n"
                        f"Max threshold: {THRESH}\n"
                        f"Res: {COLS}x{ROWS}\n"
                        f"P: {x0 + len_x * x/WIDTH} + {y0 + len_y * y/HEIGHT}i\n"
                        f"Current zoom factor: {zoom_factor}."
                    )
                if event.key == pygame.K_s:
                    name = f"mandelbrot-{random.randint(1,100000)}.png"
                    if not os.path.exists('images'):
                        os.mkdir('images')
                    file_name = f"images/{name}"
                    pygame.image.save(surf, file_name)
                    print(f"Saved image as {name} in the images directory.")
                changed = True
            
        if changed:               
            
            pixel_arr = new_pix_arr(x0,y0,x1,y1,COLS,ROWS,THRESH)
            surf = pygame.surfarray.make_surface(pixel_arr)
            display.blit(surf,(0,0))
            pygame.display.update()
        
        changed = False
        
        
if __name__ == "__main__":
    main()
