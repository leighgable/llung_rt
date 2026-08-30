import torch
import torch.nn as nn

class ContinualLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU()
        )
        # Structural placeholder. We'll map our custom Rust logic to this node name.
        self.continual_layer = nn.Linear(16, 16, bias=False) 

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.continual_layer(x)
        return x

def export():
    model = ContinualLearningModel()
    model.eval()

    # Create dummy tensor representing our batch input (Batch Size, Features)
    dummy_input = torch.randn(1, 10)

    # Dynamic axes allow the Rust runtime to accept varying batch sizes
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=17, # Modern opset matching Tract's current parsing standards [1]
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("✅ Model successfully exported to 'model.onnx'")

if __name__ == "__main__":
    export()

