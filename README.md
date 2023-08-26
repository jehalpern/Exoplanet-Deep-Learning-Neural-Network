# Exoplanet-Deep-Learning-Neural-Network
A convolutional Nerual Netwrok that classifies light curves from stellar bodies as having an exoplanet or not.



# Convolutional Neural Network for classification of Threshold Crossing Events

## Summary
This deep learning model was designed to take in light curves of stars to classify whether or not there is an exoplanet orbiting
the body. Light curves are a type of time series that plots total flux (electrons per second) over time. When an exoplanet's orbital path crosses between the star and the observer, there is a drop in flux which returns to its previous value after the orbital period is over.


![Light Curve via lightkurve documentation. Author Nicholas Saunders; STSci](http://docs.lightkurve.org/_images/tutorials_3-science-examples_exoplanets-identifying-transiting-planet-signals_29_0.png)
*Light Curve via lightkurve documentation. Author Nicholas Saunders; STSci

The purpose of this project was intended for my own learning and ultimately just a fun activity that I found addictively interesting.
In the context of learning and fun, the project was extremely successful. As is noted in the results and steps for improvement section,
the project has critical errors and is incomplete despite the fact the model was fit and trained. Many portions of this README.md may over explain concepts and portions of the project as this document serves more to solidify my own learning and understanding.



## Table of contents
I. Dependencies<br>
II. Architecture<br>
III. Results<br>
IV. Steps for improvement<br>


## Dependencies
Data Wrangling:
- astroquery
- pandas
-astropy

Data Preprocessing:
- lightkurve
- tensorflow
- tensorflow.keras
- os

Main Module:
- tensorflow
- tensorflow.keras (datasets, models, layers)
- numpy

## Architecture
#### Data Wrangling
Pre-Data module:<br>
The first sections of the pre-data module is used to gather data in the form of .fits files. Using the NASA exoplanet archive, I downloaded two csv's listing stars with and without exoplanets. The list of stars without exoplanets was queried from the "Mission Stars + ExoCat" archive (https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=mission_exocat) and the list of stars with confirmed exoplanets came from the "Planetary Systems Composite Data" archive (https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars) <br>

The actual query method utilized astroquery library and the list of stars' respective right ascension and declination values as the parameters.
#### Data Preprocessing
Extracting the light curve:<br>
The fits files have HDU's (head and data units) that contain the necessary data for extracting light curves. HDU[1] contains all the data necessary including flux and time values along with their respective error values.

> <ins>Possible Critical Error 1 - SAP FLUX</ins>: The Kepler datasets include SAP Flux (Simple Aperture Photometric) which is the calibrated flux value. The calibration is the result of a summation of pixel values minus the background contamination. The fits files I retrieved did not have these calibrated values. (Here is a NASA - TESS link on Aperture Photometry which explains the process https://heasarc.gsfc.nasa.gov/docs/tess/Aperture-Photometry-Tutorial.html) <br> <br><ins>Solution</ins>: The data['Flux'] values are a [5,5] array and so is the data['FLUX_BKG_ERR']. I used numpy's array summation of the flux and background flux error arrays in the following block of code:<br> flux_sum = np.sum(data['FLUX'], axis=(1, 2)) - np.sum(data['FLUX_BKG_ERR'], axis=(1, 2))<br>
An additional concern is addressing the identification of the optimal aperture, as I am not familiar with how the  data['FLUX_BKG_ERR'] is derived. Finding the correct aperture will key to discerning possible contamination of data and leading to a false positive.



Folding the light curve:<br>
The extracted light curve is then folded using the lightkurve's method FoldedLightCurve(). Once the light curves have been folded, the identifying labels are stripped from the plots and then written to jpeg files.

> <ins>Possible Critical Error 2 - Folding</ins>: I created two separate code blocks that fold the given light curves. Each method output a different folded light curve for the same data. Approach 1 was able to handle all steps programmatically but possibly incorrect, and approach two is more likely to be correct but involved the manual adjust of a specific value.<br>
 <ins>Approach 1</ins>:<br>The one used as the final product is the FoldedLightCurve() which has a parameter time. I passed the data['Time'] as the time argument, however, I'm not familiar with how it successfully is deriving the proper frequency and period.
 <br><ins>Approach 2</ins>:<br>In the original code block, I utilized LombScragle() along with autopower() to create the to find the frequency and power arrays, then used the frequency to calculate period and passed the period to fold(). The issue with this method was I had to manually adjust the max frequency passed to autopower() for each light curve.


> <ins>Possible Critical Error 3 - Dataset Selection</ins>: The stellar objects selected for not having exoplanets, may not necessarily have a flux output that is periodic. Therefore the folding of a light curve with no period will lead to invalid data.

TensorFlow Records:<br>
TensorFlow recommends serializing training data into TFRecord format. To make this process reusable in the future I created a separate module that can take in different parameters including the desired image dimensions.

Loading and normalizing TFRecords in Main Module:<br>
TensorFlow makes parsing, decoding, and mapping easy with their built in methods. The only additional data processing step was to normalize the pixel data from the 255 scale to a 0 to 1 scale.

#### Feature Extraction
<ins>Conv layer:</ins>
input shape=(32, 32, 3)
filters: 32
filter sample size: (3,3)
activation function: rectifier linear unit
input shape: (32, 32, 3) for first layer, and subsequent layer inputs will be the outputs of the previous layers so no shape is required.<br>

<ins>Pooling layer:</ins>
2X2 sample size with a stride of 2
#### Dense Layers
These layers (aka fully connected layers) will work to map the features outputed by the conv stack to the image classes defined in the dataset.<br>
##### Layers:

<ins>Flatten:</ins> flatten all the pixel data from the final conv layer into a 1 dimmensional sequence of data allowing the first dense layer to work with the data

<ins>First dense layer:</ins> Will perform classification by using 64 neurons that will connect features to classes and add weights to each feature and class connection. Those weights will be fed into the activation function (in this case relu) which will decide whether to activate that neuron or not.

<ins>Second dense layer:</ins> Will connect all the previous 64 neurons to the final 2 neurons(one for each class). Once again, the activation function will determine whether to activate those connections or not.

#### Training
After the dense layers are processed, there will be a training step. The compiler specifies the process for each training iteration, while the model fit calls the number of epoch's (training iterations) using the specified data.<br>
<ins>Model fit:</ins><br>
epoch: 10 training iterations<br>
back propagation: during each epoch the loss function is applied to each neuron outputing a gradient which will be fed to the optimizer.<br>
<ins>Compile:</ins><br>
Optimizer (adam): the optimizer takes the gradients calculated during back propagation to update the model parameters. It will take the first and second order moments of the gradients to adjust the learning rate of each parameter.<br>
<ins>Loss (Sparse Categorical Cross Entropy):</ins> <br>After the optimizer adjusts the parameters, the model is run again in the next epoch and the results are fed to the loss function for recaluation. These steps are repeated for each epoch.

## Results
The accuracy of the model so far is .5647, and given that it is classifying light curves between two classes, this is a very low accuracy. The project is still unfinished, but has been very fun, interesting, and exciting.
## Steps for improvement
The next steps traditionally would be to implement hyperparameter tuning. However, I strongly believe that the critical errors in the pre data processing stages need to be rectified first before attempted to fit a different model. Below are the critical errors including some additional.<br>

> <ins>Possible Critical Error 1 - SAP FLUX</ins>: The Kepler datasets include SAP Flux (Simple Aperture Photometric) which is the calibrated flux value. The calibration is the result of a summation of pixel values minus the background contamination. The fits files I retrieved did not have these calibrated values. (Here is a NASA - TESS link on Aperture Photometry which explains the process https://heasarc.gsfc.nasa.gov/docs/tess/Aperture-Photometry-Tutorial.html) <br> <br><ins>Solution</ins>: The data['Flux'] values are a [5,5] array and so is the data['FLUX_BKG_ERR']. I used numpy's array summation of the flux and background flux error arrays in the following block of code:<br> flux_sum = np.sum(data['FLUX'], axis=(1, 2)) - np.sum(data['FLUX_BKG_ERR'], axis=(1, 2))<br>
An additional concern is addressing the identification of the optimal aperture, as I am not familiar with how the  data['FLUX_BKG_ERR'] is derived. Finding the correct aperture will key to discerning possible contamination of data and leading to a false positive.

> <ins>Possible Critical Error 2 - Folding</ins>: I created two separate code blocks that fold the given light curves. Each method output a different folded light curve for the same data. Approach 1 was able to handle all steps programmatically but possibly incorrect, and approach two is more likely to be correct but involved the manual adjust of a specific value.<br>
 <ins>Approach 1</ins>:<br>The one used as the final product is the FoldedLightCurve() which has a parameter time. I passed the data['Time'] as the time argument, however, I'm not familiar with how it successfully is deriving the proper frequency and period.
 <br><ins>Approach 2</ins>:<br>In the original code block, I utilized LombScragle() along with autopower() to create the to find the frequency and power arrays, then used the frequency to calculate period and passed the period to fold(). The issue with this method was I had to manually adjust the max frequency passed to autopower() for each light curve.


> <ins>Possible Critical Error 3 - Dataset Selection</ins>: The stellar objects selected for not having exoplanets, may not necessarily have a flux output that is periodic. Therefore the folding of a light curve with no period will lead to invalid data.

> <ins>Possible Critical Error 4 - Dataset Size</ins>: Each class of light curve only has about 500 images. These are critically low values for a dataset. There is also an imbalance of data with 500 for stars without planets and 300 for stars with planets<br>
<ins>Optimal Solution</ins>: Gather a larger dataset.<br>
<ins>Alternative Solution</ins>: Implement data augmentation techniques.<br>
