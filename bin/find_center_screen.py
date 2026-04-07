"""
Run before taking PR measures. 
Presents cross at center of the screen. 
Use to align PR to center of screen.
Esc or space key will exit. Otherwise, 30s timeout.
"""
import argparse
from psychopy import core, visual, event

# Specify screen size in pixels from command line
# Will detect automatically if window is fullscr=True
parser = argparse.ArgumentParser()
parser.add_argument('--screenpixw', default=2560, type=int)
parser.add_argument('--screenpixh', default=1600, type=int)
args = parser.parse_args()

# Create window
win = visual.Window(size=(args.screenpixw,args.screenpixh),units='pix', fullscr=True)
# Create cross
centercross = visual.ShapeStim(
    win=win, vertices='cross',
    size=(win.size[1]/30, win.size[1]/30), pos=(0, 0),
    lineColor='white', fillColor='white')
# Present
centercross.draw()
win.flip()
# Wait up to 30 seconds before ending script if no key input
keys = event.waitKeys(maxWait=30.0, keyList=['escape', 'space'])
if keys:
    core.quit()