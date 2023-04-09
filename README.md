# encs6161_shared
This is the github project for ENCS 6161 of Concordia University. 
The group members are Zunao Hu and Qian Sun.

# Original Images
Both images are used for notebook file "encs_scenario1_to_2.5.ipynb".

While in other python files, only "street_view_10_cut.jpg" is used because the corresponding grayscale image will be generated in the python files.


# Python Files

encs_scenario1_to_2.5.ipynb:
This is the notebook file where the code for Scenario 1(grayscale image segmentation using k-means with histogram), Scenario 2(grayscale image segmentation
using k-means without histogram) and Scenario2.5(color image segmentation using k-means) is located.

main.py: 
The main.py file generated automatically by PyCharm.

fitfunction.py: 
The file that defines the fit function of GMM models and uses EM algorithm to achieve convergence with the selection of parameters.

encs_scenario3_gray_gmm.py: 
The file that uses the fit function from fitfunction.py to segment grayscale images by using GMM and EM.

encs_scenario3_gray_gmm_compare.py:
The file that uses the fit function from sklearn.mixture library to segment grayscale images by using GMM and EM.

encs_scenario4_color_gmm.py:
The file that uses the fit function from fitfunction.py to segment grayscale images by using GMM and EM.

encs_scenario4_color_gmm_compare.py:
The file that uses the fit function from sklearn.mixture library to segment color images by using GMM and EM.

encs_scenario5_2D_gmm.py:
The file that uses the fit function from fitfunction.py to segment 2D data by using GMM and EM.
