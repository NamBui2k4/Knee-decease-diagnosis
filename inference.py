from model import CNN
import torch
import torch.nn as nn
import cv2
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from torchvision.models import resnet18, mobilenet_v2, efficientnet_b3
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Inference CNN Model")
    parser.add_argument("--video-path", "-p", default="video/9003815L_V1.mp4", help="path to input_video file")
    parser.add_argument('--image-size', '-i', type=int, default=380, help='size of image')
    parser.add_argument("--checkpoint-path", "-c", type=str, help="Path to trained checkpoint",
                        default="checkpoint/best.pt")
    return parser.parse_args()


def inference(arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['Không mắc bệnh (normal)', 'có dấu hiệu thoái hóa (doubtful)', 'thoái hóa nhẹ (mild)',
               'thoái hóa vừa phải (moderate)', 'thoái hóa nghiêm trọng (severe)',
               'thiếu xương (osteopenia)', 'loãng xương (Osteoporosis)']
    num_class = len(classes)

    # Load the checkpoint
    checkpoint = torch.load(arg.checkpoint_path, map_location=device)

    # Initialize the model
    model = CNN(num_class).to(device)
    # model1 = resnet18(pretrained=True).to(device)
    # model2 = mobilenet_v2(pretrained=True).to(device)


    model.eval()
    
    model.load_state_dict(checkpoint['model_state'])


    # Initialize video
    cap = cv2.VideoCapture(arg.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_video = cv2.VideoWriter("output.mp4",
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                int(cap.get(cv2.CAP_PROP_FPS)),
                                (width, height))

    # Load a font that supports Vietnamese (Make sure to provide the correct path to a font file)
    font_path = "../font/arial.ttf"  # Replace with your actual font path
    font = ImageFont.truetype(font_path, size=60)

    # Video processing loop
    while cap.isOpened():
        flag, ori_frame = cap.read()
        if not flag:
            break

        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (arg.image_size, arg.image_size))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame = frame[None, :, :, :]  # Add batch dimension
        frame = torch.from_numpy(frame).float().to(device)

        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            output = model(frame)
            probs = softmax(output)

        predicted_prob, predicted_idx = torch.max(probs, dim=1)
        predicted_prob = predicted_prob.item()
        predicted_idx = predicted_idx.item()

        # Prepare text in Vietnamese
        text = f"{classes[predicted_idx]}: {predicted_prob * 100:.2f}%"

        # Convert OpenCV image (BGR) to PIL image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB))

        # Draw text on PIL image
        draw = ImageDraw.Draw(pil_image)
        draw.text((arg.image_size/2, 250), text, font=font, fill=(255, 0, 0))

        # Convert PIL image back to OpenCV format (BGR)
        ori_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out_video.write(ori_frame)

    cap.release()
    out_video.release()


if __name__ == '__main__':
    args = get_args()
    inference(args)
