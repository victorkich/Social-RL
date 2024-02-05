import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from vae.arch_torch import VAE
import torch
from skimage.transform import resize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo PyTorch VAE
vae = VAE()
vae.load_state_dict(torch.load('./vae/weights.pth', map_location=torch.device(device)))
vae.eval()  # Coloca o modelo em modo de avaliação

def preprocess_obs(obs, input_dim=(3, 64, 64)):
    # Certifique-se de que obs é um array NumPy
    if isinstance(obs, list) or obs.dtype == object:
        obs = np.array(obs.tolist(), dtype=np.float32)

    # Redimensionar a imagem para o tamanho de entrada esperado
    #if obs.shape != (input_dim[1], input_dim[2], input_dim[0]):
    #    obs = resize(obs, (input_dim[1], input_dim[2]), anti_aliasing=True)

    # Reordenar para C, H, W para PyTorch
    obs = np.transpose(obs, (2, 0, 1))
    # obs /= 255.0
    return obs

class ImageProcessor(Node):
    def __init__(self, vae_model):
        super().__init__('image_processor')
        self.vae = vae_model
        self.bridge = CvBridge()

        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.img1_reconstructed = None
        self.img2_reconstructed = None
        self.img3_reconstructed = None

        self.subscription1 = self.create_subscription(Image, 'robot1/camera/image_raw', self.image_callback1, 10)
        self.subscription2 = self.create_subscription(Image, 'robot2/camera/image_raw', self.image_callback2, 10)
        self.subscription3 = self.create_subscription(Image, 'robot3/camera/image_raw', self.image_callback3, 10)

        self.video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (192, 128))
        # width and height should be set according to your concatenated image dimensions

    def image_callback1(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.img1 = cv2.resize(cv_img, (64, 64))
        processed_img = preprocess_obs(self.img1)
        self.img1_reconstructed = self.process_image(processed_img)
        self.display_images()

    def image_callback2(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.img2 = cv2.resize(cv_img, (64, 64))
        processed_img = preprocess_obs(self.img2)
        self.img2_reconstructed = self.process_image(processed_img)
        self.display_images()

    def image_callback3(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.img3 = cv2.resize(cv_img, (64, 64))
        processed_img = preprocess_obs(self.img3)
        self.img3_reconstructed = self.process_image(processed_img)
        self.display_images()

    def process_image(self, msg):
        obs_tensor = torch.tensor([msg], dtype=torch.float32)  # Adiciona dimensão de batch
        obs_tensor /= 255.0
        vae_output, _, _ = self.vae(obs_tensor)
        vae_output = vae_output.detach().numpy()  # Detach e converta para NumPy

        # vae_output = vae_output.squeeze(0)
        vae_output = np.transpose(vae_output.squeeze(0), (1, 2, 0))
        vae_output = cv2.resize(vae_output, (64, 64)) * 255.0
        return vae_output

    def display_images(self):
        if self.img1 is not None and self.img2 is not None and self.img3 is not None:
            # Concatenate images horizontally and vertically
            concatenated_original = np.hstack((self.img1, self.img2, self.img3))
            concatenated_vae = np.hstack((self.img1_reconstructed, self.img2_reconstructed, self.img3_reconstructed))
            concatenated_final = np.vstack((concatenated_original, concatenated_vae))
            concatenated_final = cv2.resize(concatenated_final, (768, 512))
            concatenated_final = concatenated_final.astype(np.uint8)

            cv2.imshow('Processed Images', concatenated_final)
            self.video_writer.write(concatenated_final)

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
