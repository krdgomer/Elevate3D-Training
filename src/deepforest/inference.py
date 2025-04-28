from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results
# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

sample_image_path = get_data("D:/dev/projects/Bitirme-Projesi/Elevate3D-Training/src/deepforest/input.png")
boxes = model.predict_image(path=sample_image_path) 
print(boxes)
plot_results(boxes)