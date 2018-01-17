Abstract:
We consider the problem of predicting the remaining useful life (RUL) of complex engineered systems from high-dimensional raw condition monitoring data. Reliable estimation of the RUL promises improved risk management, optimized system operations, and logistic planning. Therefore, predictive modeling of system failure time is a very active research area in prognostics and health management (PHM) of complex mechatronic systems. Traditional data-driven solutions resort to steps of signal processing and feature engineering involving expert knowledge of the components degradation process and intensive labor. In this work, we adopt a residual learning strategy and propose a deep convolutional neural network (CNN) to discover high-level abstractions directly from noisy data. Raw sensor data are processed with a time window approach and feed as input to the CNN to discover the underlying mapping between the condition monitoring data and the end-of-life. The proposed method is evaluated on the popular and publicly available benchmark C-MAPSS dataset from NASA for aircraft engines. The performance of the approach is compared to other commonly applied machine learning solutions. The experimental results show that our model is able to outperform the state-of-the-art baseline methods. Therefore, it suggests that high prognostic accuracy can be achieved without prior knowledge of the degradation process with an end-to-end learning approach.

Code:
This project contains the code we use to train our three architectures, CNN, LSTM and ResNet. Our best models of each architecture were results of these scripts. In the following we describe the content of each file:
* CNN_v5_tanh_v2_0.50_HL_4x10_1.ipynb -                               script to train our CNN model

* CNN_v5_tanh_v2_0.50_HL_4x10_1.h5   -                               best CNN model achieved with CNN_v5_tanh_v2_0.50_HL_4x10_1.ipynb

* LSTM_5x42_Rearly125.ipynb         -                                script to train our LSTM model

* LSTM_125_20_42_layer.h5            -                               best LSTM model achieved with LSTM_5x42_Rearly125.ipynb

* CNN_v6_ResNet_2Stg_2xC_3xI_tanh_Drop_0.5_NoBatch_Batch_2048.ipynb - script to train our ResNet model

* CNN_v6_ResNet_2Stg_2xC_3xI_tanh_Drop_0.5_NoBatch_Batch_2048.h5   - best ResNet model achieved with CNN_v6_ResNet_2Stg_2xC_3xI_tanh_Drop_0.5_NoBatch_Batch_2048.ipynb

* RUL_FD001.txt           -                                          training data

* test_FD001.txt         -                                           test data

* train_FD001.txt           -                                        RUL of test data


The scripts can be run in Jupyter Notebook. They require following libraries to be installed:
numpy, tensorflow, keras, pandas, seaborn, matplotlib

For questions please contact Christian Schneebeli (christian dot schneebeli at bluemail.ch)
