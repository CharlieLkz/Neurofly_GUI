import tkinter as tk

root = tk.Tk()
root.title("Tkinter Test")
root.geometry("200x100")

label = tk.Label(root, text="¡Tkinter funciona!")
label.pack(pady=20)

root.mainloop()
