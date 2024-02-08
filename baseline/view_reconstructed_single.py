import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from module.vanilla_vae import VanillaVAE as VAE  # Certifique-se de que o caminho para VAE está correto

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 1024

# Carregar o modelo PyTorch VAE
vae = VAE(in_channels=3, latent_dim=latent_dim).to(device)
vae.load_state_dict(torch.load('../pretrained/vae2.pt', map_location=device))
vae.eval()  # Coloca o modelo em modo de avaliação

def preprocess_obs(obs, input_dim=(3, 64, 64)):
    obs = np.transpose(obs, (2, 0, 1))
    return obs

class ImageProcessor(Node):
    def __init__(self, vae_model):
        super().__init__('image_processor')
        self.vae = vae_model
        self.bridge = CvBridge()

        self.img_original = None  # Armazena a imagem original redimensionada
        self.img_reconstructed = None

        self.subscription = self.create_subscription(
            Image, 
            '/Robot/camera/image_raw', 
            self.image_callback, 
            10)

        self.video_writer = cv2.VideoWriter(
            '../records/vae2_output_video.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            20, 
            (256, 128))  # Ajuste a resolução para o dobro da largura da imagem individual

    def image_callback(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'rgb8')
        self.img_original = cv2.resize(cv_img, (64, 64))  # Armazena a imagem original redimensionada para exibição
        processed_img = preprocess_obs(self.img_original)
        self.img_reconstructed = self.process_image(processed_img)
        self.display_images()

    def process_image(self, img):
        obs_tensor = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        with torch.no_grad():
            vae_output, _, _, _ = self.vae(obs_tensor)
        vae_output = np.transpose(vae_output.squeeze(0).cpu().numpy(), (1, 2, 0))
        vae_output = (vae_output * 255).astype(np.uint8)
        return vae_output

    def display_images(self):
        if self.img_original is not None and self.img_reconstructed is not None:
            # Inverte os canais de RGB para BGR para ambas as imagens
            img_original_bgr = self.img_original[:, :, ::-1]
            img_reconstructed_bgr = self.img_reconstructed[:, :, ::-1]

            # Concatena as imagens original e reconstruída horizontalmente
            concatenated_image = np.hstack((img_original_bgr, img_reconstructed_bgr))
            
            # Exibe a imagem concatenada
            cv2.imshow('Original and Reconstructed Image', concatenated_image)
            self.video_writer.write(concatenated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.video_writer.release()
                cv2.destroyAllWindows()
                self.destroy_node()
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor(vae)
    rclpy.spin(image_processor)

if __name__ == '__main__':
    main()
