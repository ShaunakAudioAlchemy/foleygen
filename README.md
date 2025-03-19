# foleygen
## Foley Sound Generation with Encodec and GPT-2

This repository contains all the necessary modules and requirements that were used to execute my Masters Thesis of 'Foley Sound Generation with Encodec and GPT-2', where I used Meta's EnCodec model for audio tokenization and a pre-trained GPT-2 for audio generation. The process involves training the model on Foley sounds in 7 different categories (Dog Bark, Footstep, Gunshot, Typing on Keyboard, Moving Motor Vehicle, Rain, Sneeze Cough) to generate novel Foley sound effects.

The dataset consists of a total of 5,496 audio files in 7 categories, and is sourced from DCASE 2023 Workshop Task 7.

## Usage

# Requirements

If using conda:

`conda env create -f environment.yml` 

If using pip:

`pip install -r requirements.txt`

# Execution

In order to run the project in Google Colab, download the repository and upload all it's contents in Google Drive. Through sequentially running the Foley_Gen.ipnyb notebook attached in the repository in your respective environment. Below are the different processes involved in the notebook, that need to be executed one step at a time:

1) Installation of the necessary packages and modules
1) Data splitting and writing of a .csv file
3) Gaussian Noise Augmentation and tokenization of raw audio files using EnCodec
4) Creation of AudioDataSet class and setting up the DataLoaders
5) Training the GPT-2 model
6) Generating audio files by category

In order to use the last saved checkpoint for generating audio (300 Epochs), download the file from here: https://drive.google.com/file/d/1r1LYawsvGY_xSmvNjsz77NTgcT8UndBt/view?usp=sharing

# Results

Below are the FAD scores for the generated audio (using VGGish model embeddings). The generated audio files are within the 'synthesized' folder of the repository
<img width="497" alt="Screenshot 2025-03-19 at 6 51 23 PM" src="https://github.com/user-attachments/assets/668bcbc4-03a9-4f86-9531-ecb9dd1ef54e" />


# References:
This project was inspired by https://github.com/mrpep/encodecgpt, where EnCodec, Masked Autoencoder and GPT-2 were used for music generation. For my project, I've implemented Foley Audio generation using `audiomentations` for data augmentation, EnCodec and a pre-trained GPT-2



