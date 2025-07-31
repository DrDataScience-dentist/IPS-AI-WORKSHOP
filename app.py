import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

st.set_page_config(page_title="Kennedy Classification", layout="centered")
st.title("🦷 Kennedy Classification Predictor")

# Load model
@st.cache_resource
def load_model():
    try:
        model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
        model.load_state_dict(torch.load("efficientnet_kennedy_weights.pth", map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise

model = load_model()

class_names = ['Kennedy Class I', 'Kennedy Class II', 'Kennedy Class III', 'Kennedy Class IV']

design_recommendations = {
    'Kennedy Class I': """📌 **Design Recommendation**
- **Major Connector**: Lingual bar or Lingual plate  
- **Minor Connector**: Connected to the denture base and clasps  
- **Direct Retainer**: RPI clasp  
- **Indirect Retainers**: Perpendicular to the terminal abutments  
- **Denture Base**: Broad denture base""",
    'Kennedy Class II': """📌 **Design Recommendation**
- **Major Connector**: Lingual bar  
- **Minor Connector**: Joined to occlusal rests and clasps  
- **Direct Retainer**: RPI or circumferential clasp  
- **Indirect Retainers**: On the opposite side of the distal extension  
- **Denture Base**: Broad coverage with tissue stops""",
    'Kennedy Class III': """📌 **Design Recommendation**
- **Major Connector**: Lingual bar (mandible) or palatal strap (maxilla)  
- **Minor Connector**: Simple, connects rests and clasps  
- **Direct Retainer**: Circumferential clasp  
- **Indirect Retainers**: Not required  
- **Denture Base**: Tooth-supported, less extension needed""",
    'Kennedy Class IV': """📌 **Design Recommendation**
- **Major Connector**: Anterior palatal strap or full palatal plate  
- **Minor Connector**: Joined to rests and indirect retainers  
- **Direct Retainer**: Embrasure or combination clasp  
- **Indirect Retainers**: As needed due to anterior span  
- **Denture Base**: Covers anterior ridge with proper esthetics"""
}

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("📤 Upload a Cast Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class_index = torch.argmax(probs, dim=1).item()
        pred_class_name = class_names[pred_class_index]

    st.subheader(f"🧠 Predicted Class: {pred_class_name}")
    st.markdown(design_recommendations[pred_class_name])
