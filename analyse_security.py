import cv2
import numpy as np
import math
import scipy.stats as stats
from skimage.metrics import structural_similarity as ssim
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
import os
import struct
import matplotlib.pyplot as plt
import json

class VideoSecurityAnalyzer:
    @staticmethod
    def calculate_mse(img1, img2):
        """
        Calcule l'erreur quadratique moyenne (Mean Squared Error)
        """
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    @staticmethod
    def calculate_psnr(mse):
        """
        Calcule le pic du rapport signal/bruit (Peak Signal-to-Noise Ratio)
        """
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * math.log10(max_pixel / math.sqrt(mse))

    @staticmethod
    def calculate_ssim(img1, img2):
        """
        Calcule l'index de similarité structurelle (Structural Similarity Index)
        """
        # Convert to grayscale for faster SSIM calculation
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2, win_size=7)

    @staticmethod
    def calculate_entropy(image):
        """
        Calcule l'entropie de l'image
        """
        # Aplatir l'image et calculer l'histogramme
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        
        # Normaliser l'histogramme
        probabilities = hist / np.sum(hist)
        
        # Calculer l'entropie
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probabilities])
        
        return entropy

    @staticmethod
    def calculate_correlation(image):
        """
        Calcule la corrélation entre les canaux de l'image
        """
        def safe_correlation(x, y):
            """Calculate correlation with handling for constant arrays"""
            # Check if either array is constant
            if np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return np.corrcoef(x, y)[0, 1]
        
        # Aplatir les canaux de l'image
        r_channel = image[:,:,0].flatten()
        g_channel = image[:,:,1].flatten()
        b_channel = image[:,:,2].flatten()
        
        # Calculer les corrélations
        corr_rg = safe_correlation(r_channel, g_channel)
        corr_rb = safe_correlation(r_channel, b_channel)
        corr_gb = safe_correlation(g_channel, b_channel)
        
        return corr_rg, corr_rb, corr_gb

    def plot_histogram(self, original_video_path, encrypted_video_path, output_path, sample_frames=5):
        """
        Analyse et trace les histogrammes des frames originales et chiffrées
        """
        # Ouvrir les vidéos
        cap_orig = cv2.VideoCapture(original_video_path)
        cap_enc = cv2.VideoCapture(encrypted_video_path)
        
        # Obtenir le nombre total de frames
        total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = total_frames // sample_frames
        
        # Créer une figure avec une grille de sous-plots
        fig, axes = plt.subplots(sample_frames, 2, figsize=(15, 5*sample_frames))
        fig.suptitle('Analyse des Histogrammes: Original vs Chiffré', fontsize=16)
        
        for i in range(sample_frames):
            # Positionner à la frame souhaitée
            frame_pos = i * frame_interval
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            cap_enc.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Lire les frames
            ret_orig, frame_orig = cap_orig.read()
            ret_enc, frame_enc = cap_enc.read()
            
            if not (ret_orig and ret_enc):
                break
                
            # Tracer l'histogramme original
            axes[i,0].set_title(f'Frame {frame_pos} - Original')
            for j, color in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([frame_orig], [j], None, [256], [0, 256])
                axes[i,0].plot(hist, color=color, label=color.upper())
            axes[i,0].set_xlabel('Valeurs de Pixel')
            axes[i,0].set_ylabel('Fréquence')
            axes[i,0].legend()
            
            # Tracer l'histogramme chiffré
            axes[i,1].set_title(f'Frame {frame_pos} - Chiffré')
            for j, color in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([frame_enc], [j], None, [256], [0, 256])
                axes[i,1].plot(hist, color=color, label=color.upper())
            axes[i,1].set_xlabel('Valeurs de Pixel')
            axes[i,1].set_ylabel('Fréquence')
            axes[i,1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Libérer les ressources
        cap_orig.release()
        cap_enc.release()

    def analyze_histogram_statistics(self, original_video_path, encrypted_video_path, sample_rate=30):
        """
        Analyse statistique des histogrammes
        """
        # Ouvrir les vidéos
        cap_orig = cv2.VideoCapture(original_video_path)
        cap_enc = cv2.VideoCapture(encrypted_video_path)
        
        # Préparer les conteneurs pour les statistiques
        orig_stats = {'mean': [], 'std': [], 'skewness': [], 'kurtosis': []}
        enc_stats = {'mean': [], 'std': [], 'skewness': [], 'kurtosis': []}
        
        frame_count = 0
        
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_enc, frame_enc = cap_enc.read()
            
            if not (ret_orig and ret_enc):
                break
                
            frame_count += 1
            if frame_count % sample_rate != 0:
                continue
                
            # Calculer les statistiques pour chaque canal
            for frame, stats in [(frame_orig, orig_stats), (frame_enc, enc_stats)]:
                for channel in cv2.split(frame):
                    # Calculer les moments statistiques
                    mean = np.mean(channel)
                    std = np.std(channel)
                    skewness = np.mean(((channel - mean)/std)**3) if std != 0 else 0
                    kurtosis = np.mean(((channel - mean)/std)**4) if std != 0 else 0
                    
                    stats['mean'].append(mean)
                    stats['std'].append(std)
                    stats['skewness'].append(skewness)
                    stats['kurtosis'].append(kurtosis)
        
        # Libérer les ressources
        cap_orig.release()
        cap_enc.release()
        
        # Calculer les moyennes des statistiques
        rapport = {
            "Statistiques Histogramme Original": {
                "Moyenne": float(np.mean(orig_stats['mean'])),
                "Écart-type": float(np.mean(orig_stats['std'])),
                "Asymétrie": float(np.mean(orig_stats['skewness'])),
                "Kurtosis": float(np.mean(orig_stats['kurtosis']))
            },
            "Statistiques Histogramme Chiffré": {
                "Moyenne": float(np.mean(enc_stats['mean'])),
                "Écart-type": float(np.mean(enc_stats['std'])),
                "Asymétrie": float(np.mean(enc_stats['skewness'])),
                "Kurtosis": float(np.mean(enc_stats['kurtosis']))
            }
        }
        
        return rapport

    def analyze_encryption(self, original_video_path, encrypted_video_path, decrypted_video_path, sample_rate=30):
        """
        Analyse complète de la sécurité du chiffrement avec échantillonnage
        """
        # Ouvrir les vidéos
        original_cap = cv2.VideoCapture(original_video_path)
        encrypted_cap = cv2.VideoCapture(encrypted_video_path)
        decrypted_cap = cv2.VideoCapture(decrypted_video_path)
        
        # Obtenir le nombre total de frames
        total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        
        # Préparer les listes de résultats
        mse_results = []
        psnr_results = []
        ssim_results = []
        entropy_original = []
        entropy_encrypted = []
        correlation_original = []
        correlation_encrypted = []
        
        frame_count = 0
        
        print(f"Analysing video with {total_frames} frames, sampling every {sample_rate} frames")
        
        while True:
            ret_orig, orig_frame = original_cap.read()
            ret_enc, frame_enc = encrypted_cap.read()
            ret_dec, dec_frame = decrypted_cap.read()
            
            if not (ret_orig and ret_enc and ret_dec):
                break
                
            frame_count += 1
            
            # Skip frames according to sample rate
            if frame_count % sample_rate != 0:
                continue
                
            processed_frames += 1
            
            # Afficher la progression
            progress = (frame_count * 100) / total_frames
            print(f"\rProgress: {progress:.1f}% - Processed {processed_frames} samples", end="")
            
            # Calculs des métriques
            mse = self.calculate_mse(orig_frame, dec_frame)
            psnr = self.calculate_psnr(mse)
            ssim_score = self.calculate_ssim(orig_frame, dec_frame)
            
            # Entropie
            entropy_orig = self.calculate_entropy(orig_frame)
            entropy_enc = self.calculate_entropy(frame_enc)
            
            # Corrélation
            corr_orig = self.calculate_correlation(orig_frame)
            corr_enc = self.calculate_correlation(frame_enc)
            
            # Stocker les résultats
            mse_results.append(mse)
            psnr_results.append(psnr)
            ssim_results.append(ssim_score)
            entropy_original.append(entropy_orig)
            entropy_encrypted.append(entropy_enc)
            correlation_original.append(corr_orig)
            correlation_encrypted.append(corr_enc)
        
        print("\nAnalysis complete!")
        
        # Fermer les captures
        original_cap.release()
        encrypted_cap.release()
        decrypted_cap.release()
        
        # Générer un rapport
        rapport = {
            "Informations Generales": {
                "Nombre total de frames": total_frames,
                "Frames analysees": processed_frames,
                "Taux d'echantillonnage": f"1/{sample_rate}"
            },
            "Metriques Moyennes": {
                "MSE Moyen": float(np.mean(mse_results)),
                "PSNR Moyen": float(np.mean(psnr_results)),
                "SSIM Moyen": float(np.mean(ssim_results))
            },
            "Entropie": {
                "Entropie Moyenne (Originale)": float(np.mean(entropy_original)),
                "Entropie Moyenne (Chiffree)": float(np.mean(entropy_encrypted))
            },
            "Correlation": {
                "Correlation Moyenne (Originale)": [float(x) for x in np.mean(correlation_original, axis=0)],
                "Correlation Moyenne (Chiffree)": [float(x) for x in np.mean(correlation_encrypted, axis=0)]
            }
        }
        
        return rapport

# Example usage
if __name__ == "__main__":
    # Create analyzer
    security_analyzer = VideoSecurityAnalyzer()
    
    # Paths
    input_video = 'University Promo Video Template (Editable).mp4'
    encrypted_video = 'encrypted_video.mp4'
    decrypted_video = 'decrypted_video.mp4'
    histogram_output = 'histogrammes_analyse.png'
    
    # Analyze encryption
    print("Analyzing encryption...")
    rapport_securite = security_analyzer.analyze_encryption(
        input_video, 
        encrypted_video, 
        decrypted_video,
        sample_rate=30
    )
    
    # Generate histograms
    print("\nGenerating histograms...")
    security_analyzer.plot_histogram(
        input_video,
        encrypted_video,
        histogram_output,
        sample_frames=5
    )
    
    # Analyze histogram statistics
    print("\nAnalyzing histogram statistics...")
    hist_stats = security_analyzer.analyze_histogram_statistics(
        input_video,
        encrypted_video,
        sample_rate=30
    )
    
    # Print reports
    print("\nRapport de sécurité:")
    print(json.dumps(rapport_securite, indent=2, ensure_ascii=False))
    
    print("\nStatistiques des histogrammes:")
    print(json.dumps(hist_stats, indent=2, ensure_ascii=False))