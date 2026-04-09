"""
Presents colors (input RGB definitions as tsv file) and takes spectrum measurements
with spectroradiometer. Saves out entire spectra into csv file. 
"""
import argparse 
import pandas as pd
import csv
import sys
from psychopy import core, visual, hardware, event
from psychopy_photoresearch.pr import PR655

event.globalKeys.add(key='q', func=core.quit) # to quit at any time point

parser = argparse.ArgumentParser()
parser.add_argument('--in_rgb', required=True, help='txt with rgb rows')
parser.add_argument('--out_csv', required=True, help='output csv')
parser.add_argument('--photometer', required=True, help='PR655 or PR670')
parser.add_argument('--port', default='/dev/cu.usbmodem1101', type=str, help='serial port PR is plugged into') #'/dev/cu.usbmodem1101'
parser.add_argument('--waittime', type=float, default=3.0, help='time between color pres and measurement started')
parser.add_argument('--reps', type=int, default=1, help='how many measures to take per color')
parser.add_argument('--screenpixw', default=2560, type=int)
parser.add_argument('--screenpixh', default=1600, type=int)
parser.add_argument('--on_timer', default=False, type=bool)
parser.add-argument('--time_between' default = 20., type=float)
args = parser.parse_args()

if args.photometer == 'PR655':
    phot = PR655(args.port)
elif args.photometer == 'PR670':
    phot = PR670(args.port)
else:
    print('input PR not recognized')
    sys.exit()

# Read in rgb definitions
rgb_defs = pd.read_csv(args.in_rgb, sep='\t')

# Initialize window
win = visual.Window(size=(args.screenpixw,args.screenpixh), units='pix', color=(50,50,50), colorSpace='rgb255', fullscr=True)
# Initialize stim
colorstim = visual.Rect(win, width=win.size[0], height= win.size[1], 
            fillColor=(0,0,0),colorSpace='rgb255')
            
# Start PR remote mode
phot.startRemoteMode()

# Start clock; used if measurements are being taken strictly on timer
cycle = core.Clock()
# Run measurements
with open(args.out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['rep','id', 'r','g','b','nm','power'])
    # For however many times you want to measure each color
    for rep in range(args.reps):
        # Go through all colors
        for i in range(len(rgb_defs)):
            color_rgb = rgb_defs.iloc[i]
            # To handle colors that are out of gamut
            if color_rgb['R']<0. or color_rgb['G']<0. or color_rgb['B']<0.:
                colorstim.fillColor = [0.,0.,0.]
                colorstim.draw()
                win.flip()
                cycle.reset()
                w.writerow([rep, color_rgb['ID'],color_rgb['R'],color_rgb['G'],color_rgb['B'], -1., -1.])
                # Pad to overall total time if working on timer rather than direct measurements
                remaining = args.time_between - cycle.getTime()
                if args.on_timer == True:
                    if remaining > 0:
                    core.wait(remaining)
                    else:
                        print('ERROR: color measurements became out of sync during presentation of color ', str(color_rgb['ID']), ' please increase argument time_between')
                        win.close()
                        core.quit()
            # If color def is valid
            else:
                colorstim.fillColor = [color_rgb['R'],color_rgb['G'],color_rgb['B']]
                colorstim.draw()
                win.flip()
                cycle.reset()
                core.wait(args.waittime) # Wait _ seconds after presentation before measurement
                
                phot.measure(timeOut=8.0) # If measurement takes longer than 8 seconds, something is probably wrong
                spec = phot.getLastSpectrum(parse=True)
                wavelengths, powers = spec # separate the arrays in spec
                
                # Plot cartoon of spectrum (visual feedback so you know it looks reasonable)
                # Normalize x and then scale and flip so Long->Short left to right
                x = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
                x = (x - 0.5) * (win.size[0] * -0.3) 
                y = (powers - powers.min()) / (powers.max() - powers.min())
                y = (y - 0.5) * (win.size[0] * 0.1) 

                plot = visual.ShapeStim(
                    win,
                    vertices=list(zip(x, y)),
                    lineColor="white",
                    lineWidth=2,
                    closeShape=False)

                win.flip()
                plot.draw()
                win.flip()
                core.wait(1.0) # Plot spectrum cartoon for 1 s
                
                # Write all info to row in output csv
                for nm, power in zip(wavelengths, powers):
                    w.writerow([rep, color_rgb['ID'],color_rgb['R'],color_rgb['G'],color_rgb['B'], float(nm), float(power)])
                    
                # Pad to overall total time if working on timer rather than direct measurements
                remaining = args.time_between - cycle.getTime()
                if args.on_timer == True:
                    if remaining > 0:
                    core.wait(remaining)
                    else:
                        print('ERROR: color measurements became out of sync during presentation of color ', str(color_rgb['ID']), ' please increase argument time_between')
                        win.close()
                        core.quit()
win.close()
core.quit()