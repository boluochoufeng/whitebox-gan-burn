import torch
import torchvision.models as models
from torchvision.models.vgg import VGG, VGG19_Weights


def export_vgg19_weights(output_path: str, output_index=25):
    vgg: VGG = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    vgg.eval()

    features_state_dict = vgg.features.state_dict()
    layer_mapping = {
        "0": "conv1_1",
        "2": "conv1_2",
        "5": "conv2_1",
        "7": "conv2_2",
        "10": "conv3_1",
        "12": "conv3_2",
        "14": "conv3_3",
        "16": "conv3_4",
        "19": "conv4_1",
        "21": "conv4_2",
        "23": "conv4_3",
        "25": "conv4_4",
        "28": "conv5_1",
        "30": "conv5_2",
        "32": "conv5_3",
        "34": "conv5_4",
    }
    renamed_state_dict = {}
    for old_name, param in features_state_dict.items():
        parts = old_name.split(".")
        layer_index = parts[0]
        param_type = parts[1]

        if layer_index in layer_mapping:
            if int(layer_index) > output_index:
                continue
            new_name = f"{layer_mapping[layer_index]}.{param_type}"
            renamed_state_dict[new_name] = param
            print(f"mapping: {old_name} -> {new_name}, shape: {param.shape}")

    torch.save(renamed_state_dict, output_path)
    print(f"Done. Export {len(renamed_state_dict)} params.")

def main():
    export_vgg19_weights("./vgg19.pth")


if __name__ == "__main__":
    main()

