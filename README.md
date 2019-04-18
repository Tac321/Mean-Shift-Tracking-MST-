# MST: Mean Shift Tracking
Here shown is an (MST)  solution for tracking images. This is useful for low quality images of a discinct intensity/texture/color.


 # Note
This MST Algorithm identifies objects which are distinguishable due to textutre, and/or intensity. The code draws a rectangular patch about target images and as the target image pans in the camera view, the MST target rectangle follows the target image. 

If the code is looking at a flat gradient image (sky, wall, etc...) a search box will grow in size to select bigger or lower resolution target images. Of the objects detected, the one the closest to the center of the camera will be highlighted cyan.

If the center of the cyan targeting rectangle is close enough to the center of the camera for a duration of time, the MST tracker will turn on and only that target image will be highlighted in the image. To stop tracking a target image, aim the ccamera at a flat intensity image, like a dark room, wall, or place your hand blocking the camera from receiving light.


## How to run

### Run code
1) Open libIAT_MST_SCD_Master.py
2) Press F5   or the play button on the top  of the screen.
3) To end the program, press "Esc" key

## Example
<img src="https://github.com/Tac321/Mean-Shift-Tracking-MST-/blob/master/MST_Pic.png" width="700" />
