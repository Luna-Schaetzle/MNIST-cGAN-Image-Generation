import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os

# ===========================
# 1. Generator laden
# ===========================

def load_generator_model(model_path='generator_cgan.h5'):
    if not os.path.exists(model_path):
        messagebox.showerror("Modell nicht gefunden", f"Das Generator-Modell wurde nicht gefunden unter:\n{model_path}")
        return None
    try:
        generator = load_model(model_path)
        return generator
    except Exception as e:
        messagebox.showerror("Fehler beim Laden des Modells", str(e))
        return None

# ===========================
# 2. Bewertungsmodell laden
# ===========================

def load_evaluation_model(model_path='evaluation_model.h5'):
    if not os.path.exists(model_path):
        messagebox.showerror("Bewertungsmodell nicht gefunden", f"Das Bewertungsmodell wurde nicht gefunden unter:\n{model_path}")
        return None
    try:
        evaluator = load_model(model_path)
        return evaluator
    except Exception as e:
        messagebox.showerror("Fehler beim Laden des Bewertungsmodells", str(e))
        return None

# ===========================
# 3. Bild generieren
# ===========================

def generate_image(generator, digit, evaluator, noise_dim=100, num_generations=5):
    best_score = -1
    best_image = None

    for _ in range(num_generations):
        # Rauschvektor erzeugen
        noise = tf.random.normal([1, noise_dim])
        
        # Label-Vektor erstellen (One-Hot)
        label = tf.one_hot([digit], depth=10)
        
        # Bild generieren
        generated_image = generator([noise, label], training=False)
        
        # Denormalisieren von [-1,1] zu [0,255]
        generated_image = (generated_image[0, :, :, 0] * 127.5 + 127.5).numpy().astype(np.uint8)
        
        # Bild für Bewertung vorbereiten
        img_array = generated_image.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Form: (1, 28, 28, 1)
        
        # Bewertung durchführen
        predictions = evaluator.predict(img_array)
        score = np.max(predictions)  # Höchste Wahrscheinlichkeit
        
        # Bestes Bild auswählen
        if score > best_score:
            best_score = score
            best_image = Image.fromarray(generated_image, mode='L')

    return best_image, best_score

# ===========================
# 4. GUI erstellen
# ===========================

class CGANApp:
    def __init__(self, root, generator, evaluator):
        self.root = root
        self.generator = generator
        self.evaluator = evaluator
        self.root.title("MNIST cGAN Generator mit Bewertung")
        self.root.geometry("600x900")
        self.root.resizable(False, False)
        
        # Stil konfigurieren
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Titel
        self.title_label = ttk.Label(root, text="MNIST cGAN Generator mit Bewertung", font=("Helvetica", 16))
        self.title_label.pack(pady=10)
        
        # Auswahl der Ziffer
        self.digit_label = ttk.Label(root, text="Wähle eine Ziffer (0-9):")
        self.digit_label.pack(pady=5)
        
        self.digit_var = tk.IntVar()
        self.digit_var.set(0)
        self.digit_spinbox = ttk.Spinbox(root, from_=0, to=9, textvariable=self.digit_var, width=5)
        self.digit_spinbox.pack(pady=5)
        
        # Anzahl der Generierungen
        self.num_label = ttk.Label(root, text="Anzahl der Generierungen pro Ziffer:")
        self.num_label.pack(pady=5)
        
        self.num_var = tk.IntVar()
        self.num_var.set(5)
        self.num_spinbox = ttk.Spinbox(root, from_=1, to=20, textvariable=self.num_var, width=5)
        self.num_spinbox.pack(pady=5)
        
        # Generieren-Button
        self.generate_button = ttk.Button(root, text="Bild generieren", command=self.display_generated_image)
        self.generate_button.pack(pady=10)
        
        # Bewertungsanzeige
        self.score_label = ttk.Label(root, text="Bewertung: N/A")
        self.score_label.pack(pady=5)
        
        # Bildanzeige
        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)
        
        # Speichern-Button
        self.save_button = ttk.Button(root, text="Bild speichern", command=self.save_image, state='disabled')
        self.save_button.pack(pady=5)
        
        # Referenz zum aktuellen Bild
        self.current_image = None
    
    def display_generated_image(self):
        digit = self.digit_var.get()
        num_generations = self.num_var.get()
        img, score = generate_image(self.generator, digit, self.evaluator, num_generations=num_generations)
        if img is not None:
            self.current_image = img
            # Konvertieren für Tkinter
            imgtk = ImageTk.PhotoImage(image=img.resize((280, 280)))
            self.image_label.configure(image=imgtk)
            self.image_label.image = imgtk
            # Bewertung anzeigen
            self.score_label.config(text=f"Bewertung: {score:.4f}")
            # Aktivieren des Speichern-Buttons
            self.save_button.config(state='normal')
        else:
            messagebox.showerror("Fehler", "Kein Bild generiert.")

    def save_image(self):
        if self.current_image:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png"),
                                                               ("All files", "*.*")],
                                                    title="Bild speichern unter")
            if save_path:
                try:
                    self.current_image.save(save_path)
                    messagebox.showinfo("Erfolg", f"Bild erfolgreich gespeichert unter:\n{save_path}")
                except Exception as e:
                    messagebox.showerror("Fehler", f"Fehler beim Speichern des Bildes:\n{str(e)}")
        else:
            messagebox.showwarning("Kein Bild", "Es gibt kein Bild zum Speichern. Bitte generiere zuerst ein Bild.")

# ===========================
# 5. Hauptfunktion
# ===========================

def main():
    # Pfade zu den Modellen
    generator_path = 'generator_cgan-numbers.h5'
    evaluator_path = 'evaluation_model.h5'
    
    # Generator laden
    generator = load_generator_model(generator_path)
    if generator is None:
        return
    
    # Bewertungsmodell laden
    evaluator = load_evaluation_model(evaluator_path)
    if evaluator is None:
        return
    
    # GUI starten
    root = tk.Tk()
    app = CGANApp(root, generator, evaluator)
    root.mainloop()

# ===========================
# 6. Skript ausführen
# ===========================

if __name__ == '__main__':
    main()
