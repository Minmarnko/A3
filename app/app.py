import streamlit as st
import torch
from utils import initialize_model, greedy_decode

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = initialize_model('add')
save_path = './models/addmodel.pt'
model.load_state_dict(torch.load(save_path, map_location=device))
model = model.to(device)  # Move the model to the selected device

# Streamlit UI
def main():
    st.set_page_config(page_title="English - Burmese Translation", layout="centered")
    
    # Title and Instructions
    st.title("English - Burmese Translation")
    st.write("Enter your English sentence below to translate it into Burmese.")
    
    # User input
    prompt = st.text_input("Enter text:", "")
    
    if st.button("Submit"):
        if prompt.strip():
            # Ensure input tensor is moved to the correct device
            generation, _ = greedy_decode(model, prompt, max_len=50, device=device)
            if '<eos>' in generation:
                generation.remove('<eos>')
            sentence = ' '.join(generation)
            
            # Display Translation
            st.subheader("Translation:")
            st.write(sentence)
        else:
            st.warning("Please enter a valid input.")

if __name__ == "__main__":
    main()
