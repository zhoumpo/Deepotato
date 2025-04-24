#!/usr/bin/env python3
"""
Main application entry point for the WebSocket Server GUI.
"""
import tkinter as tk

from app.gui import MainApplication


def main():
    """Initialize and run the application."""
    # Create the root window
    root = tk.Tk()
    root.title("Deepotato")
    root.geometry("800x600")

    # Create and pack the main application
    app = MainApplication(root)
    app.pack(fill="both", expand=True)

    # Center the window on screen (optional)
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()
