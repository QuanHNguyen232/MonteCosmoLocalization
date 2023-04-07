# Anki Cozmo Kidnapping using Monte Carlo Localization

Members: Nick, Quan, Doug, and Brayton

Goal:
* Implement MCL with Anki Cozmo
* Create Panorama of collected images from which the Cozmo bot can use to find its position and relocate itself.

[Cozmo API](https://data.bit-bots.de/cozmo_sdk_doc/cozmosdk.anki.com/docs/api.html)

Helpful [link](https://github.com/nheidloff/visual-recognition-for-cozmo-with-tensorflow/blob/master/1-take-pictures/take-pictures.py) for taking pictures with cozmo and saving them.

---
## Table of Contents
1. [Setup](#setup)
   1. [Computer](#computer-setup)
   1. [Android phone](#mobile-phone-android-setup)
1. [Task](#tasks)
1. [Working log](#working-log)

---

## Setup
### Computer Setup:
1.	Install Anki Cozmo SDK
1.	Enable/install Android Debug Bridge on Android device
1.	Setup environment (recommend using Anaconda): Python 3.8 + required packages in requirements.txt:
   * Create environment:
      ```bash
      conda create -n envname python=3.8
      ```
   * Activate environment:
      ```bash
      conda activate envname
      ```
   * Install all required packages:
      ```bash
      pip install -r requirements.txt
      ```

### Mobile Phone (Android) Setup:
1.	Enable USB debugging on Android device
1.	Install Official Cozmo app on phone
1.	Run USB cable from phone to computer
1. Run `adb devices` on CommandPrompt on computer to check and authorize your phone
1.	Open Cozmo app and turn Cozmo bot on 
1.	Connect to Cozmo bot’s wifi network.
1.	Enable SDK on Cozmo app

---

## Tasks
* [ ] Change all read imgs to get 3D imgs (3 channels) -> modify convolution+pool functions (Quan)
* [ ] Recreate MCL as per supplied examples (based on [L.Attai,R.Fuller,C.Rhodes](http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/MCLocalize.py)))
   * [ ] Polish borrowed code: make more succint, use methods we have already coded. 
* [ ] Read MCL slides + Java demo code (convert to python) (Bray & Doug)
* [ ] Try running MCL w/ pano
* [ ] Future:
   * How to improve accuracy
   * How to make it works in dark environment (may use edge detection??)
   * How to localize with just 4 images (0, 90, 180, 270 degrees) (divide imgs into $n$ cols then compare w/ $n$ cols of each img for all $m$ ims?)
* [X] How to rotate robot
* [X] How to get image from cozmo’s camera (file `picture-collection.py`)
* [X] Fix MCL code for images (check [Youtube](https://www.youtube.com/watch?v=JhkxtSn9eo8) and github) (file `my-MCL.py`)
* [X] Create a panorama (optional)
   * Check openCV Stitching (e.g from prev group: [L.Attai,R.Fuller,C.Rhodes](http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/Stitching.py))
* [X] Crop pano img + sensor img in MCL algo (Nick)


---

## Working log
| Data/Time | Activity | Member |
|:-|:-|-:|
| 3/17: 1-2pm | Setup Project | Brayton & Nick |
| 3/23: 4:15-5:15pm | Setup Project & Doc of steps | Doug & Nick |
| 3/24: 1-5pm | Connect to Cozmo, implement basic MCL & pic collection | Doug, Quan & Nick |
| 3/29: 2-5pm | MCL, refine code for pic collection | Doug, Quan & Nick |
| 3/31: 2-4:45pm | Done take1Pic, kidnap | Doug, Quan & Nick |
| 3/31: 5:15-6:30pm | Modify and fix bug in MCL | Quan |
| 4/1: 1-1:50am | Add MSE+Cos_Similar, try MCL, bad result --> suggest creating pano | Quan |
| 4/5: 3:30-5:30pm  | Image stiching for creation of pano | Quan & Nick |
| 4/7: 1:20-3:30pm| Image cropping, MCL redo | Nick, Quan, Brayton |
| 4/7: 3:30-4:30 pm| MCL redo | Nick, Quan |