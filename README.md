# Camera-Calibration
Implementing two very popular camera calibration techniques: The Direct Linear Transform and Zhang's Method, and testing the quality of the parameters estimated from these calibration methods.  

If the jupyter notebook doesn't load, use the notebook viewer: **[nbviewer](https://nbviewer.jupyter.org/github/BonJovi1/Camera-Calibration/blob/master/code.ipynb)** \
Or else, check out `code.md`, because jupyter notebooks convert to markdown pretty well! 

### DLT 
The Question:
- For the given image `calib-object.jpg` using any 20-30 different points on different planes and perform the Direct Linear Transform (DLT) based calibration. Report the projection matrix, camera matrix, rotation matrix and projection center. Note that you need to manually estimate the image co-ordinates of the given world points and refer to calib-object-legend.jpg for world measurements. Each chessblock is 28X28 mm.
- Implement the RANSAC based variant of the above calibration method.
- Repeat the above experiments after correcting for radial distortion. Estimate the radial distortion parameters from the straight lines in the image. 

### Zhang's Method
The Question:
- Use checkerboard images IMG5456.JPG - IMG5470.JPG and perform camera calibration using Zhang’s Method. 
- Using the estimated camera parameters compute the image points and overlay a wire-frame over the actual image of chessboard using straight lines between the computed points. 
- What is the image of the world origin, given the calibration matrix? Does this result bear out in your observations?

### Hands-On
- Perform the above calibration methods using the images taken by your camera. Use the calibration object for which you can measure the world co-ordinates for DLT and printed checkerboard pattern for Zhangs Method.
- Vary the focus of your phone as well. 
