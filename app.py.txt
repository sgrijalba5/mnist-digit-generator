import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Modelo autoencoder condicional (igual al entrenamiento)
class ConditionalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        label_onehot = F.one_hot(labels, num_classes=10).float()
        x_cat = torch.cat((x, label_onehot), dim=1)
        encoded = self.encoder(x_cat)
        encoded_cat = torch.cat((encoded, label_onehot), dim=1)
        decoded = self.decoder(encoded_cat)
        return decoded.view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    model = ConditionalAutoencoder()
    model.load_state_dict(torch.load("conditional_autoencoder.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Interfaz
st.title("🧠 MNIST Digit Generator")
digit = st.selectbox("Select a digit (0–9):", list(range(10)))
generate = st.button("Generate 5 Images")

if generate:
    with st.spinner("Generating images..."):
        labels = torch.tensor([digit]*5)
        noise = torch.rand((5, 1, 28, 28))  # entrada aleatoria
        generated = model(noise, labels).detach().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(generated[i][0], cmap="gray")
            axes[i].axis("off")
        st.pyplot(fig)
