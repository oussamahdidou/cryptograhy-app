import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding  
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
import os
import struct
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
class VideoEncryptor:
    def __init__(self):
        # Générer une clé privée ECC
        self.ecc_private_key = ec.generate_private_key(
            ec.SECP256R1(), 
            backend=default_backend()
        )
        self.ecc_public_key = self.ecc_private_key.public_key()
        self.display_keys()
    def generate_aes_key(self):
        """Génère une clé AES de 256 bits"""
        return os.urandom(32)
    def display_keys(self):
        # Sérialiser et afficher la clé privée
        private_pem = self.ecc_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        # Sérialiser et afficher la clé publique
        public_pem = self.ecc_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Afficher les clés
        print("Private ecc Key:")
        print(private_pem.decode())
        print("Public ecc Key:")
        print(public_pem.decode())

    def frame_to_bytes(self, frame):
        """Convertit un frame en bytes avec des informations de contexte"""
        height, width, channels = frame.shape
        frame_bytes = frame.tobytes()
        # Ajouter des métadonnées à l'en-tête
        header = struct.pack('III', height, width, channels)
        return header + frame_bytes

    def bytes_to_frame(self, frame_bytes):
        """Reconvertit des bytes en frame à partir des métadonnées"""
        # Extraire les métadonnées
        height, width, channels = struct.unpack('III', frame_bytes[:12])
        frame_data = frame_bytes[12:]
        
        # Reconstruire le frame
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        return frame.reshape(height, width, channels)

    def encrypt_video(self, input_path, output_path):
        """
        Chiffre une vidéo frame par frame avec AES
        """
        # Générer la clé AES
        aes_key = self.generate_aes_key()

        # Ouvrir la vidéo source
        cap = cv2.VideoCapture(input_path)
        
        # Obtenir les propriétés de la vidéo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialiser le writer pour la vidéo chiffrée
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        encrypted_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir le frame en bytes avec métadonnées
            frame_bytes = self.frame_to_bytes(frame)
            
            # Chiffrer le frame avec AES
            iv = os.urandom(16)
            encryptor = Cipher(
                algorithms.AES(aes_key), 
                modes.CBC(iv),  
                backend=default_backend()
            ).encryptor()
            
            # Padding PKCS7 
            padder = padding.PKCS7(256).padder()  # Correction ici
            padded_data = padder.update(frame_bytes) + padder.finalize()
            
            encrypted_frame_bytes = encryptor.update(padded_data) + encryptor.finalize()
            
            # Écrire le frame chiffré (en noir, car on ne peut pas écrire des données chiffrées)
            out.write(np.zeros((height, width, 3), dtype=np.uint8))
            
            encrypted_frames.append((iv, encrypted_frame_bytes))
        
        cap.release()
        out.release()
        
        # Chiffrer la clé AES avec ECC
        encrypted_aes_key = self.encrypt_aes_key(aes_key)
        
        return encrypted_aes_key, encrypted_frames, (width, height, fps) , self.ecc_private_key , self.ecc_public_key

    def encrypt_aes_key(self, aes_key):
        """
        Chiffre la clé AES avec la clé publique ECC
        """
        shared_key = self.ecc_private_key.exchange(
            ec.ECDH(), 
            self.ecc_public_key
        )
        # Utiliser HKDF correctement
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'ecc encryption',
            backend=default_backend()
        ).derive(shared_key)
        
        # Chiffrer la clé AES
        iv = os.urandom(16)
        encryptor = Cipher(
            algorithms.AES(derived_key), 
            modes.CBC(iv), 
            backend=default_backend()
        ).encryptor()
        
        # Padding PKCS7 pour la clé
        padder = padding.PKCS7(256).padder()  # Correction ici
        padded_aes_key = padder.update(aes_key) + padder.finalize()
        
        encrypted_aes_key = encryptor.update(padded_aes_key) + encryptor.finalize()
        
        return iv, encrypted_aes_key

    def decrypt_video(self, encrypted_aes_key, encrypted_frames, output_path, video_params, 
                      ecc_private_key=None):
        """
        Déchiffre une vidéo chiffrée
        """
        # Si aucune clé privée n'est fournie, utiliser la clé générée
        if ecc_private_key is None:
            ecc_private_key = self.ecc_private_key
        
        # Déchiffrer la clé AES
        iv_ecc, encrypted_key = encrypted_aes_key
        shared_key = ecc_private_key.exchange(
            ec.ECDH(), 
            ecc_private_key.public_key()
        )
        
        # Utiliser HKDF correctement
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'ecc encryption',
            backend=default_backend()
        ).derive(shared_key)
        
        decryptor_ecc = Cipher(
            algorithms.AES(derived_key), 
            modes.CBC(iv_ecc), 
            backend=default_backend()
        ).decryptor()
        
        # Dépadding de la clé AES
        unpadder = padding.PKCS7(256).unpadder()  # Correction ici
        decrypted_key = decryptor_ecc.update(encrypted_key) + decryptor_ecc.finalize()
        aes_key = unpadder.update(decrypted_key) + unpadder.finalize()
        
        # Récupérer les paramètres vidéo
        width, height, fps = video_params
        
        # Initialiser le writer pour la vidéo déchiffrée
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Déchiffrer chaque frame
        for iv, encrypted_frame_bytes in encrypted_frames:
            decryptor = Cipher(
                algorithms.AES(aes_key), 
                modes.CBC(iv), 
                backend=default_backend()
            ).decryptor()
            
            # Déchiffrer avec unpadding
            decrypted_padded_frame_bytes = decryptor.update(encrypted_frame_bytes) + decryptor.finalize()
            unpadder = padding.PKCS7(256).unpadder()  # Correction ici
            decrypted_frame_bytes = unpadder.update(decrypted_padded_frame_bytes) + unpadder.finalize()
            
            # Reconvertir les bytes en frame
            decrypted_frame = self.bytes_to_frame(decrypted_frame_bytes)
            
            out.write(decrypted_frame)
        
        out.release()

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un encrypteur
    video_encryptor = VideoEncryptor()
    
    # Chiffrer une vidéo
    encrypted_aes_key, encrypted_frames, video_params = video_encryptor.encrypt_video(
        'University Promo Video Template (Editable).mp4', 
        'encrypted_video.mp4'
    )
    
    # Déchiffrer la vidéo
    video_encryptor.decrypt_video(
        encrypted_aes_key, 
        encrypted_frames, 
        'decrypted_video.mp4',
        video_params
    )