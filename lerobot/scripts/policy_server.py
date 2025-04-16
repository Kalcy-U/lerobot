from flask import Flask, request, jsonify
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import torch
import numpy as np

app = Flask(__name__)

def create_policy_server(policy, device, use_amp):
    """
    Create a Flask server to handle remote policy predictions.
    
    Args:
        policy: The policy model for action selection
        device: Torch device (cpu or cuda)
        use_amp: Whether to use automatic mixed precision
    """
    
    @app.route('/predict', methods=['POST'])
    def predict_action_server():
        # print("hello\n")
        try:
            # Get observation from request
            observation = request.get_json()
            
            # Convert numpy arrays back to tensors
            for name in observation:
                observation[name] = torch.from_numpy(np.array(observation[name]))
                
                if "image" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()
                observation[name] = observation[name].unsqueeze(0)
                observation[name] = observation[name].to(device)
            
            # Compute action
            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
            ):
                action = policy.select_action(observation)
                action = action.squeeze(0).to("cpu").numpy()
            
            # print("end\n")
            
            return jsonify({"action": action.tolist()})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app

policy = PI0Policy.from_pretrained("/home/fdse/.cache/modelscope/hub/models/lerobot/pi0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True
app = create_policy_server(policy, device, use_amp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)