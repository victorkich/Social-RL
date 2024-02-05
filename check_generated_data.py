import cv2
import numpy as np
import os

def load_and_display_images(directory='./data/rollout/'):
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            
            # Supondo que as imagens estejam armazenadas com a chave 'obs'
            # e cada observação esteja no formato (altura, largura, canais)
            images = data['obs']
            
            for image in images:
                # Convertendo se necessário, dependendo de como as imagens foram salvas
                # Se as imagens já estiverem no formato correto para visualização, você pode remover esta linha
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                cv2.imshow('Image', image)
                key = cv2.waitKey(0)  # Espera até que uma tecla seja pressionada para mostrar a próxima imagem
                
                if key == 27:  # ESC key para sair
                    cv2.destroyAllWindows()
                    return

if __name__ == '__main__':
    load_and_display_images()
