"""
GUI module that contains the main application window and UI components.
"""

import re
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk

from app.websocket_client import WebSocketClient
from app.websocket_server import WebSocketServer


class ConsoleOutput(scrolledtext.ScrolledText):
    """A text widget that acts as a console for displaying output."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Dark console with light text
        self.configure(
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1E1F22",  # Dark background
            fg="#A9B7C6",  # Light text
            insertbackground="#A9B7C6",  # Cursor color
            selectbackground="#214283",  # Selection background
            selectforeground="#FFFFFF",  # Selection text
            state="disabled",
        )

    def write(self, text):
        """Write text to the console."""
        self.configure(state="normal")
        self.insert(tk.END, text)
        self.see(tk.END)  # Auto-scroll to the bottom
        self.configure(state="disabled")

    def clear(self):
        """Clear all text from the console."""
        self.configure(state="normal")
        self.delete(1.0, tk.END)
        self.configure(state="disabled")


class MainApplication(ttk.Frame):
    """Main application container."""

    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.websocket_server = None
        self.websocket_client = None

        # Create UI elements
        self.create_widgets()
        self.create_layout()

        # Add sample welcome message to console
        self.console.write("Deepotato Server Started\n")

    def create_widgets(self):
        """Create all the widgets for the application."""
        # Create a frame for the console
        self.console_frame = ttk.LabelFrame(self, text="Console Output")
        self.console = ConsoleOutput(self.console_frame, width=80, height=20)

        # Create a frame for the controls
        self.control_frame = ttk.LabelFrame(self, text="Server Controls")

        # Port input field
        self.port_frame = ttk.Frame(self.control_frame)
        self.port_label = ttk.Label(self.port_frame, text="Port:")
        self.port_var = tk.StringVar(value="8765")
        self.port_entry = ttk.Entry(
            self.port_frame, textvariable=self.port_var, width=10
        )

        # Server control buttons
        self.button_frame = ttk.Frame(self.control_frame)
        self.start_button = ttk.Button(
            self.button_frame, text="Start Server", command=self.start_server
        )
        self.stop_button = ttk.Button(
            self.button_frame,
            text="Stop Server",
            command=self.stop_server,
            state="disabled",
        )
        self.test_client_button = ttk.Button(
            self.button_frame, text="Test Client", command=self.test_client
        )
        self.clear_console_button = ttk.Button(
            self.button_frame, text="Clear Console", command=self.console.clear
        )

        # Server status indicator
        self.status_frame = ttk.Frame(self.control_frame)
        self.status_label = ttk.Label(self.status_frame, text="Server Status:")
        self.status_value = ttk.Label(
            self.status_frame, text="Stopped", foreground="red"
        )

    def create_layout(self):
        """Arrange the widgets in the application."""
        # Main layout
        self.console_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.console.pack(fill="both", expand=True, padx=5, pady=5)

        self.control_frame.pack(fill="x", padx=10, pady=5)

        # Layout for port input
        self.port_frame.pack(fill="x", padx=5, pady=5)
        self.port_label.pack(side="left", padx=5)
        self.port_entry.pack(side="left", padx=5)

        # Layout for buttons
        self.button_frame.pack(fill="x", padx=5, pady=5)
        self.start_button.pack(side="left", padx=5)
        self.stop_button.pack(side="left", padx=5)
        self.test_client_button.pack(side="left", padx=5)
        self.clear_console_button.pack(side="right", padx=5)

        # Layout for status indicator
        self.status_frame.pack(fill="x", padx=5, pady=5)
        self.status_label.pack(side="left", padx=5)
        self.status_value.pack(side="left", padx=5)

    def start_server(self):
        """Start the WebSocket server."""
        port_str = self.port_var.get().strip()

        # Validate port input
        if not re.match(r"^\d+$", port_str):
            messagebox.showerror("Invalid Port", "Port must be a number.")
            return

        port = int(port_str)
        if port < 1024 or port > 65535:
            messagebox.showerror("Invalid Port", "Port must be between 1024 and 65535.")
            return

        try:
            # Create and start the WebSocket server
            self.websocket_server = WebSocketServer(
                port, console_callback=self.console.write
            )
            success = self.websocket_server.start()

            # Wait briefly to ensure server has time to start
            self.parent.after(500, self._check_server_status)

        except Exception as e:
            self.console.write(f"Error starting server: {str(e)}\n")
            self.websocket_server = None

    def _check_server_status(self):
        """Check if the server is actually running and update UI accordingly."""
        if self.websocket_server and self.websocket_server.running:
            self.console.write(
                f"WebSocket server is running on port {self.port_var.get()}\n"
            )
            self.update_server_status(True)
        else:
            self.console.write("Failed to start WebSocket server\n")
            self.websocket_server = None
            self.update_server_status(False)

    def stop_server(self):
        """Stop the WebSocket server."""
        if self.websocket_server:
            try:
                self.websocket_server.stop()
                self.console.write("WebSocket server stopped\n")
                self.websocket_server = None
                self.update_server_status(False)
            except Exception as e:
                self.console.write(f"Error stopping server: {str(e)}\n")

    def update_server_status(self, is_running):
        """Update the server status indicator and button states."""
        if is_running:
            self.status_value.config(text="Running", foreground="green")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.test_client_button.config(state="normal")
        else:
            self.status_value.config(text="Stopped", foreground="red")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.test_client_button.config(state="disabled")

    def test_client(self):
        """Open a dialog to get a test message and send it to the server."""
        if not self.websocket_server:
            messagebox.showerror(
                "Server Not Running", "Please start the server before testing."
            )
            return

        if not self.websocket_server.running:
            messagebox.showerror(
                "Server Not Running",
                "Server is not in running state. Please restart the server.",
            )
            return

        # Get the message to send
        message = simpledialog.askstring(
            "Test Client", "Enter message to send to the server:", parent=self.parent
        )

        if not message:  # User canceled or entered empty string
            return

        # Get current port
        port = int(self.port_var.get().strip())

        # Create a client and send the message
        try:
            self.console.write(
                f"Starting test client to send message to localhost:{port}\n"
            )
            self.websocket_client = WebSocketClient(
                "localhost", port, self.console.write
            )
            success = self.websocket_client.send_test_message(message)

            if not success:
                self.console.write("Failed to send test message\n")

        except Exception as e:
            self.console.write(f"Error in test client: {str(e)}\n")
