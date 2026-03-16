# imagen_local.py (versión corregida solo del panel)
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

class DrawPanel:
    def draw(self, size=(28,28), line_width=3, scale=10):
        w, h = int(size[0]*scale), int(size[1]*scale)
        lw = int(line_width*scale)

        self.root = tk.Tk()
        self.root.title("Dibuja un dígito y cierra la ventana")

        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="white")
        self.canvas.pack()

        # Imagen donde se guarda el dibujo
        self.image = Image.new("L", (w, h), 255)
        self.draw_img = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None

        def on_press(event):
            self.last_x, self.last_y = event.x, event.y

        def on_move(event):
            if self.last_x is not None and self.last_y is not None:
                self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                        width=lw, fill="black", capstyle=tk.ROUND, smooth=True)
                self.draw_img.line((self.last_x, self.last_y, event.x, event.y),
                                   fill=0, width=lw)
            self.last_x, self.last_y = event.x, event.y

        def on_release(event):
            self.last_x, self.last_y = None, None

        self.canvas.bind("<ButtonPress-1>", on_press)
        self.canvas.bind("<B1-Motion>", on_move)
        self.canvas.bind("<ButtonRelease-1>", on_release)

        tk.Label(self.root, text="Dibuja con el mouse y cerrá la ventana para continuar.").pack()
        self.root.mainloop()

        # Escala al tamaño deseado
        img_small = self.image.resize(size, Image.Resampling.LANCZOS)
        return img_small

    from PIL import ImageChops, ImageFilter

    import numpy as np

    def preprocess_mnist(pil_img):
        # 1. Asegurar modo L
        img = pil_img.convert('L')
    
        # 2. Invertir (MNIST fondo negro)
        img = ImageOps.invert(img)
    
        # 3. Filtrar para engrosar trazos (dilatación)
        img = img.filter(ImageFilter.MaxFilter(3))
    
        # 4. Centrar por centro de masa
        arr = np.array(img)
        total = arr.sum()
        if total > 0:
            coords = np.indices(arr.shape)
            cy = (coords[0] * arr).sum() / total
            cx = (coords[1] * arr).sum() / total
            shift_x = int(np.round(arr.shape[1]/2 - cx))
            shift_y = int(np.round(arr.shape[0]/2 - cy))
            img = ImageChops.offset(img, shift_x, shift_y)
    
        # 5. Normalizar a [0,1] y reshape
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        return arr, img
    

