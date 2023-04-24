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
1. [Tasks](#tasks)
1. [Files and Dirs](#files-and-dirs)
1. [Working log](#working-log)

---

## Setup

Following the [installation guide from Cozmo](http://cozmosdk.anki.com/docs/initial.html):
   1. [Initial setup](http://cozmosdk.anki.com/docs/initial.html):
      * Encourage using anaconda/miniconda instead of directly installing Python 3.x.
      * SDK Example Programs: download and extract to any directory
   1. Installation:
      * On phone:
         * Install Official Cozmo app ([Android](https://play.google.com/store/apps/details?id=com.digitaldreamlabs.cozmo2&hl=en_US&gl=US)/[iOS](https://apps.apple.com/us/app/cozmo/id1154282030))
      * On computer:
         * Install [Android Debug Bridge](http://cozmosdk.anki.com/docs/adb.html#adb), follow that to [Final installation](http://cozmosdk.anki.com/docs/adb.html#final-install). The “Enable USB debugging” step may vary based on your phone, so check the [android instruction](https://developer.android.com/studio/debug/dev-options) if needed.
   1. Installing Python:
      * In Anaconda Powershell Prompt, create new environment for developing Cozmo bot using this command: “conda create -n envname python=3.8”
      * Activate environment and install required libraries: “conda activate envname” then “pip install cozmo[camera].”
   1. Final step: download the SDK Examples, open Cozmo app, connect with cozmo bot (ensure the robot does not fall off the desk) -> setting -> enable SDK. Then try running the examples.

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
1.	Install Official Cozmo app on phone ([Android](https://play.google.com/store/apps/details?id=com.digitaldreamlabs.cozmo2&hl=en_US&gl=US)/[iOS](https://apps.apple.com/us/app/cozmo/id1154282030))
1.	Run USB cable from phone to computer
1. Run `adb devices` on CommandPrompt on computer to check and authorize your phone. (unil it appears **device** instead of **unauthorized**)
1.	Open Cozmo app and turn Cozmo bot on 
1.	Connect to Cozmo bot’s wifi network.
1.	Enable SDK on Cozmo app

<p align="right"><a href="#anki-cozmo-kidnapping-using-monte-carlo-localization">[Back to top]</a></p>

---

## Tasks
* [ ] Get robot to turn to location it has highest belief prob after MCL done
* [ ] Change all read imgs to get 3D imgs (3 channels) -> modify convolution+pool functions (Quan)
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
* [X] Recreate MCL as per supplied examples (based on [L.Attai,R.Fuller,C.Rhodes](http://cs.gettysburg.edu/~tneller/archive/cs371/cozmo/22sp/fuller/MCLocalize.py)))

<p align="right"><a href="#anki-cozmo-kidnapping-using-monte-carlo-localization">[Back to top]</a></p>

---
## Files and Dirs
* `data/`: data directory collected from MCL, used to create histogram.
* `cozmo-images-kidnap/`: images for kidnap problem (collected data)
* `cozmo-imgs-data1/`: data for remote work
* `Histogram.py`: histogram of probs from MCL
* `cozmo_MCL.py`: New MCL implementation, based on previous group's work
* `img_processing.py`:
   * get imgs func
   * process imgs
   * save imgs
* `pic_collection.py`:
   * collection of pictures for pano
   * taking of single image for MCL
* `MCL_old.py`: old MCL (not true MCL -> not accurate)

<p align="right"><a href="#anki-cozmo-kidnapping-using-monte-carlo-localization">[Back to top]</a></p>


---

## Working log
<details>
<summary>Click to expand</summary>

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
| 4/12: 2:20-4:50pm| Working on new MCL based on previous group efforts | Nick |
| 4/12: 7:00-7:50pm| MCL debugging | Nick |
| 4/14: 1:00-2:30PM| MCL testing/debugging | Brayton |
| 4/14: 1:00-5:20PM| MCL testing/debugging | Nick |
| 4/16: 9:20-10:00PM| Make Cozmo relocalize after MCL | Nick |
| 4/16: 2:00-4:30PM| Make Cozmo relocalize after MCL | Nick | 
| 4/16: 2:00-4:30PM| Make Cozmo relocalize after MCL | Nick | 
| 4/21: 1:00-2:30PM| Cozmo localization tuning, website with documentation | Nick, Doug, Brayton |
| 4/21: 2:30-5PM| Cozmo localization tuning, bins for histogram | Nick |

</details>
<p align="right"><a href="#anki-cozmo-kidnapping-using-monte-carlo-localization">[Back to top]</a></p>

---

**Convert `md` to `html`**: use this [codebeautify.org](https://codebeautify.org/markdown-to-html) for conversion.

<p align="right"><a href="#anki-cozmo-kidnapping-using-monte-carlo-localization">[Back to top]</a></p>