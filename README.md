# vocal 🔥

Take a look at main.ipynb!

## Project Overview
The objective of our project was to extract vocals from one song and accompaniment from another, and then merge the two. Future improvements might include pitch tuning and denoising. The project primarily leverages a pre-trained Spleeter model by Deezer, with additional operations performed using the powerful audio manipulation library, Librosa.

## Audio Representation in Computers
Each sound at a given instant can be represented as a sum of various frequencies, each with its own intensity. The sample rate is the number of frequencies we can distinguish. When we convert a song into a numerical array, the result is an array of shape (sample rate x number of time units). Each value in this array represents the intensity of a specific frequency at a given moment. The concept can be visualized through a spectrogram as shown below:
![image](https://github.com/songs-merger/ml-experiments/assets/78561567/01239647-807e-4cf3-a94c-e7eaf6288dac)


## Introduction to Spleeter
Spleeter is powered by U-net, an encoder-decoder architecture that employs convolutional neural networks instead of fully connected layers. We use transposed filters for upscaling, and the model incorporates several other strategies, such as skip connections from encoder outputs to the decoder, batch normalization in the encoder, 0.5 dropout in the decoder, and the use of LeakyReLU and ELU instead of ReLUs. Spleeter consists of 12 layers in total (6 in the encoder, 6 in the decoder).
Source code: https://github.com/deezer/spleeter/blob/master/spleeter/model/functions/unet.py
Here's a diagram to give you an intuition of how U-net works:

![image](https://github.com/songs-merger/ml-experiments/assets/78561567/8926c0d9-4c0f-42f8-a27d-8a7f9dfc9c01)

## Loudness Normalization
To ensure that the merged tracks have a balanced volume, we split both songs into frames and compute the average intensity for each frequency in each frame. We have experimented with two approaches: adjusting vocal intensity all at once or by individual frames. It appears that modifying the intensity all at once yields superior results.

## Denoising (WIP)
Denoising was implemented using Short-Time Fourier Transform (STFT). The process involves converting the audio signal into the frequency domain and creating a binary mask that identifies whether the intensity at a given frequency exceeds a certain threshold. If it does, it is kept; otherwise, it is set to zero, thereby reducing noise.
Denoising is not used in the final version though, as it produces some super weird sounds.

## Morphing (WIP)
Morphing is the process of blending two audio clips in a way that results in a seamless transition between them. We attempted to use Deep Convolutional Generative Adversarial Networks (DCGAN) for this task, but found that it required prohibitive amounts of time to train.

## Beat Alignment (WIP Beat Alignment 2.0 using nn)
Beat alignment ensures that the beats of the vocals and accompaniment are synchronized. The align_beat function detects the beat pattern in each track and calculates the necessary adjustments to align them. It then modifies the audio clips accordingly, and the final outputs are trimmed to a common length.
