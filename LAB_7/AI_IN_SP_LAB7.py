#!/usr/bin/env python
# coding: utf-8

# In[1]:



pip install librosa hmmlearn


# In[10]:



import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from hmmlearn import hmm


# In[11]:


pip install numpy==1.21


# In[13]:


import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from hmmlearn import hmm

def load_audio(file_path):
    ydata, sr = librosa.load(file_path, sr=None)
    return ydata, sr

def stft_features(ydata, sr):
    stft = np.abs(librosa.stft(ydata))
    return stft

def train_hmm(features, n_components=3, n_iter=100):
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
    model.fit(features)
    return model

def plot_stft(stft, sr):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT')
    plt.show()

def classify_signal(model, features):
    # Predict using the trained HMM model
    labels = model.predict(features.T)  # Transpose features to fit HMM's requirement
    return labels

def main():
    audio_file_path ="AI_IN_SP_AUDIO_RECORDING.wav"
    
    # Load audio
    y, sr = load_audio(audio_file_path)
    
    # Extract STFT features
    stft = stft_features(y, sr)
    
    # Plot STFT
    plot_stft(stft, sr)
    
    # Train HMM
    model = train_hmm(stft.T)  # Transpose stft to fit HMM's requirement
    
    # Classify signal using trained HMM
    labels = classify_signal(model, stft)
    
    # Plot the classification result
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(labels)), labels, label='Classified State')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('HMM Classification Result')
    plt.legend()
    plt.show()

    # Print trained model parameters
    print("HMM Model Parameters:")
    print("Transition Matrix:")
    print(model.transmat_)
    print("Means:")
    print(model.means_)
    print("Covariances:")
    print(model.covars_)
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[9]:


import numpy as np
import librosa
from hmmlearn import hmm
 
# Load the audio file
file_path = "AISP_VOICE_21007.wav"  # Assuming the file is in the same directory as your notebook
signal, sr = librosa.load(file_path, sr=None)
 
# Extract STFT features
stft = np.abs(librosa.stft(signal))
 
# Convert STFT features to observation sequence
obs_seq = np.transpose(stft)
 
# Train an HMM model
num_states = 3  # Number of states in the HMM
num_mix = 1     # Number of mixtures in each state
model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=1000)
model.fit(obs_seq)
 
# Classify using trained HMM model
predicted_labels = model.predict(obs_seq)
 
# Output the predicted labels
print("Predicted labels:", predicted_labels)


# In[ ]:




