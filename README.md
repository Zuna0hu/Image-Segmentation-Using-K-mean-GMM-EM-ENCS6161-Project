# ENCS 6161 Winter 2023 Group Project
This is the github project for ENCS 6161 of Concordia University. 
The group members are Zunao Hu and Qian Sun.

# Descreption of the Project
This research explores the effectiveness of the K-means algorithm and Gaussian Mixture Model (GMM) with Expectation Maximization (EM) in segmenting images. The K-means algorithm was applied to grayscale and color images with and without intensity histograms. A customized function was developed to fit data to Gaussian Mixture and was used for grayscale and color image segmentation. The results were compared with the Gaussian Mixture class from the sklearn library. Additionally, the function was applied to fit Gaussian Mixture to 2-D datasets and compared with input signals. This study presents a comprehensive analysis of the advantages and limitations of K-means and GMM-EM for image segmentation.

<div style="display: flex; justify-content: center;">
  <img src="https://user-images.githubusercontent.com/124393973/235541331-cfe42593-5034-464b-9584-767c23391e9c.png" alt="k_means_gray_with_histogram" width="480" height="240">
</div>

<div style="display: flex; justify-content: center;">
  <img src="https://user-images.githubusercontent.com/124393973/235541510-3e3096eb-69e7-4f80-82f3-478bf4b93d07.png" alt="k_means_gray_without_histogram" width="480" height="240"/>
</div>

<div style="display: flex; justify-content: center;">
 <img src="https://user-images.githubusercontent.com/124393973/235541592-0f80d54d-e208-45e7-b761-c1948207d120.png" alt="k_means_color" width="480" height="240"/>
</div>

<div style="display: flex; justify-content: center;">
  <img src="https://user-images.githubusercontent.com/124393973/235541691-e2466b13-be36-4b20-96e6-127d8634424f.jpg" alt="gmm_grayscale" width="240"/>
  <img src="https://user-images.githubusercontent.com/124393973/235541749-49b94106-7c97-4e20-9f8b-bdf40c423b14.jpg" alt="gmm_color" width="240"/>
</div>
<div style="display: flex; justify-content: center;">
  <img src="https://user-images.githubusercontent.com/124393973/235541810-231a4c1f-cca2-4fee-932a-2c85eddf3ed3.png" alt="gmm_2d_orig" width="280" height="240"/>
  <img src="https://user-images.githubusercontent.com/124393973/235541892-64fbae9e-44c7-40e2-b9dd-049cb31d0c88.png" alt="gmm_2d_seg" width="280" height="240"/>
</div>


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
