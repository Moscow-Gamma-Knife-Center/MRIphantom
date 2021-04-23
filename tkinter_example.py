import tkinter as tk

class Demo1:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.buttonloadCT = tk.Button(self.frame, text = 'Load CT', width = 25, command = self.loadCT())
        self.buttonloadCT.pack()
        self.buttonloadMR = tk.Button(self.frame, text='Load MR', width=25, command=self.loadMR())
        self.buttonloadMR.pack()
        self.frame.pack()
    def loadCT(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)

    def loadMR(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)

class Demo2:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.quitButton = tk.Button(self.frame, text = 'Quit', width = 25, command = self.close_windows)
        self.quitButton.pack()
        self.frame.pack()
    def close_windows(self):
        self.master.destroy()

def main():
    root = tk.Tk()
    app = Demo1(root)
    root.mainloop()

main()