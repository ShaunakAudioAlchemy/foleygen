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

# Results

FAD Scores:

<img width="532" alt="Screenshot 2024-12-02 at 9 01 23 PM" src="https://github.com/user-attachments/assets/a5746b3e-700d-44ed-b720-5285ef7bf9fb">

# References:
EnCodec: A state-of-the-art high fidelity neural audio codec.



