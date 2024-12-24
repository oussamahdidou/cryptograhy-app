import tkinter as tk
from tkinter import Tk, ttk, filedialog, messagebox
import json
from encryption import VideoEncryptor
from analyse_security import VideoSecurityAnalyzer
import threading
import os
from cryptography.hazmat.primitives import serialization
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
class VideoEncryptionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Encryption Tool")
        self.root.geometry("800x600")
        
        # Initialize encryption objects
        self.video_encryptor = VideoEncryptor()
        self.security_analyzer = VideoSecurityAnalyzer()
        self.encrypted_frames=[]
        # Store paths
        self.input_video_path = tk.StringVar()
        self.encrypted_video_path = tk.StringVar()
        self.decrypted_video_path = tk.StringVar()
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Create tabs
        self.encryption_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.encryption_tab, text='Encryption/Decryption')
        self.notebook.add(self.analysis_tab, text='Security Analysis')
        
        self._create_encryption_tab()
        self._create_analysis_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    def _create_encryption_tab(self):
        # Input video section
        input_frame = ttk.LabelFrame(self.encryption_tab, text="Input Video", padding="5")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Select video:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(input_frame, textvariable=self.input_video_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Browse", command=self._browse_input).pack(side=tk.LEFT, padx=5)
        
        # Actions frame
        actions_frame = ttk.LabelFrame(self.encryption_tab, text="Actions", padding="5")
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Encrypt Video", command=self._start_encryption).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Decrypt Video", command=self._start_decryption).pack(side=tk.LEFT, padx=5)
        
        # Output paths frame
        output_frame = ttk.LabelFrame(self.encryption_tab, text="Output Paths", padding="5")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(output_frame, text="Encrypted:").pack(anchor=tk.W)
        ttk.Entry(output_frame, textvariable=self.encrypted_video_path, width=70).pack(fill=tk.X, padx=5)
        
        ttk.Label(output_frame, text="Decrypted:").pack(anchor=tk.W)
        ttk.Entry(output_frame, textvariable=self.decrypted_video_path, width=70).pack(fill=tk.X, padx=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(self.encryption_tab, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for log
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _create_analysis_tab(self):
        # Analysis controls frame
        controls_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Controls", padding="5")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Sample Rate:").pack(side=tk.LEFT, padx=5)
        self.sample_rate = ttk.Entry(controls_frame, width=10)
        self.sample_rate.insert(0, "30")
        self.sample_rate.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Run Analysis", command=self._start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Generate Histograms", command=self._generate_histograms).pack(side=tk.LEFT, padx=5)
        
        # Analysis results frame
        results_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

    def _browse_input(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.input_video_path.set(filename)
            # Auto-generate output paths
            base_dir = os.path.dirname(filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self.encrypted_video_path.set(os.path.join(base_dir, f"{base_name}_encrypted.mp4"))
            self.decrypted_video_path.set(os.path.join(base_dir, f"{base_name}_decrypted.mp4"))

    def _log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def _start_encryption(self):
        if not self.input_video_path.get():
            messagebox.showerror("Error", "Please select an input video first!")
            return
            
        self.status_var.set("Encrypting video...")
        self._log_message("Starting encryption process...")

        def encrypt():
            try:
                encrypted_aes_key, encrypted_frames, video_params, ecc_private , ecc_public_key = self.video_encryptor.encrypt_video(
                    self.input_video_path.get(),
                    self.encrypted_video_path.get()
                )
                self.encrypted_frames =encrypted_frames
                ecc_private_hex = ecc_private.private_numbers().private_value.to_bytes(
                (ecc_private.key_size + 7) // 8, 'big').hex()

                # Use public_bytes() to export the public key in a specific format (e.g., uncompressed, PEM, etc.)
                ecc_public_hex = ecc_public_key.public_bytes(
                    encoding=serialization.Encoding.X962,
                    format=serialization.PublicFormat.UncompressedPoint
                ).hex()                # Save encryption data
                encryption_data = {
                    'ecc_private_Key':ecc_private_hex,
                    'ecc_public_Key':ecc_public_hex,
                    'encrypted_aes_key': (encrypted_aes_key[0].hex(), encrypted_aes_key[1].hex()),
                    'video_params': video_params
                }
                
                data_path = self.encrypted_video_path.get() + '.json'
                with open(data_path, 'w') as f:
                    json.dump(encryption_data, f)
                
                self.status_var.set("Encryption completed!")
                self._log_message("Encryption completed successfully!")
                messagebox.showinfo("Success", "Video encryption completed!")
                
            except Exception as e:
                self.status_var.set("Encryption failed!")
                self._log_message(f"Encryption failed: {str(e)}")
                messagebox.showerror("Error", f"Encryption failed: {str(e)}")
            
            finally:
                self.progress_var.set(0)
        
        threading.Thread(target=encrypt, daemon=True).start()

    def _start_decryption(self):
        if not self.encrypted_video_path.get():
            messagebox.showerror("Error", "Please specify the encrypted video path!")
            return
        try:
            # Load encryption data
            data_path = self.encrypted_video_path.get() + '.json'
            with open(data_path, 'r') as f:
                encryption_data = json.load(f)
            
            encrypted_aes_key = (
                bytes.fromhex(encryption_data['encrypted_aes_key'][0]),
                bytes.fromhex(encryption_data['encrypted_aes_key'][1])
            )
            # ecc_private_Key_hex = encryption_data['ecc_private_Key']
            video_params=encryption_data['video_params']
 
            curve = ec.SECP384R1()  # The curve used (same curve as in encryption)
            # ecc_private_key = deserialize_private_key(ecc_private_Key_hex, curve)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load encryption data: {str(e)}")
            return
            
        self.status_var.set("Decrypting video...")
        self._log_message("Starting decryption process...")
        # Function to deserialize ECC private key from hex
        def deserialize_private_key(hex_private_key, curve):
            # Convert the hex string back to bytes
            private_bytes = bytes.fromhex(hex_private_key)
            
            # Recreate the private key object
            private_key = ec.derive_private_key(
                int.from_bytes(private_bytes, 'big'), curve, default_backend()
            )
            return private_key

        def decrypt():
            try:
                self.video_encryptor.decrypt_video(
                    encrypted_aes_key,
                    self.encrypted_frames,  # This needs to be modified to handle the encrypted frames
                    self.decrypted_video_path.get(),
                    video_params,
       
                )
                
                self.status_var.set("Decryption completed!")
                self._log_message("Decryption completed successfully!")
                messagebox.showinfo("Success", "Video decryption completed!")
                
            except Exception as e:
                self.status_var.set("Decryption failed!")
                self._log_message(f"Decryption failed: {str(e)}")
                messagebox.showerror("Error", f"Decryption failed: {str(e)}")
            
            finally:
                self.progress_var.set(0)
        
        threading.Thread(target=decrypt, daemon=True).start()

    def _start_analysis(self):
        if not all([self.input_video_path.get(), self.encrypted_video_path.get(), self.decrypted_video_path.get()]):
            messagebox.showerror("Error", "Please ensure all video paths are specified!")
            return
            
        self.status_var.set("Running security analysis...")
        self._log_message("Starting security analysis...")
        
        def analyze():
            try:
                # Run analysis
                results = self.security_analyzer.analyze_encryption(
                    self.input_video_path.get(),
                    self.encrypted_video_path.get(),
                    self.decrypted_video_path.get(),
                    sample_rate=int(self.sample_rate.get())
                )
                
                # Display results
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, json.dumps(results, indent=2))
                
                self.status_var.set("Analysis completed!")
                self._log_message("Security analysis completed successfully!")
                
            except Exception as e:
                self.status_var.set("Analysis failed!")
                self._log_message(f"Analysis failed: {str(e)}")
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            
            finally:
                self.progress_var.set(0)
        
        threading.Thread(target=analyze, daemon=True).start()

    def _generate_histograms(self):
        if not all([self.input_video_path.get(), self.encrypted_video_path.get()]):
            messagebox.showerror("Error", "Please ensure input and encrypted video paths are specified!")
            return
            
        self.status_var.set("Generating histograms...")
        self._log_message("Starting histogram generation...")
        
        def generate():
            try:
                output_path =self.encrypted_video_path.get() + 'histograms_analysis.png'
                self.security_analyzer.plot_histogram(
                    self.input_video_path.get(),
                    self.encrypted_video_path.get(),
                    output_path,
                    sample_frames=5
                )
                
                self.status_var.set("Histograms generated!")
                self._log_message(f"Histograms saved to: {output_path}")
                messagebox.showinfo("Success", f"Histograms saved to: {output_path}")
                
            except Exception as e:
                self.status_var.set("Histogram generation failed!")
                self._log_message(f"Histogram generation failed: {str(e)}")
                messagebox.showerror("Error", f"Histogram generation failed: {str(e)}")
            
            finally:
                self.progress_var.set(0)
        
        threading.Thread(target=generate, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEncryptionUI(root)
    root.mainloop()