#!/usr/bin/env python3
"""
Main application entry point for Deepotato AI Bot.
"""
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, Callable
import torch  # Import torch for device detection

# Add proper path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our modules
from app.websocket_server import WebSocketServer
from app.game_agent import GameAgent

class RedirectText:
    """Redirect print statements to the UI text widget."""
    
    def __init__(self, text_widget):
        """Initialize with a text widget to redirect to."""
        self.text_widget = text_widget
        self.buffer = ""
        
    def write(self, string):
        """Write to the text widget."""
        self.buffer += string
        if '\n' in self.buffer:
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, self.buffer)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state="disabled")
            self.buffer = ""
    
    def flush(self):
        """Flush the buffer."""
        if self.buffer:
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, self.buffer)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state="disabled")
            self.buffer = ""


class DeepLearningBotUI(ttk.Frame):
    """UI for controlling the deep learning bot."""
    
    def __init__(self, root):
        """Initialize the UI."""
        super().__init__(root)
        self.root = root
        self.root.title("Deepotato - Deep Learning Bot Controller")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create UI
        self.create_widgets()
        self.pack(fill=tk.BOTH, expand=True)
        
        # Initialize server and agent
        self.server = None
        self.agent = None
        self.update_timer = None
        self.game_state = None
        
        # Start the UI update timer
        self.root.after(1000, self.update_ui)
    
    def create_widgets(self):
        """Create all UI widgets."""
        # Create the main frame with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Server controls
        server_frame = ttk.LabelFrame(main_frame, text="WebSocket Server", padding="5")
        server_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Server controls - row 1
        ttk.Label(server_frame, text="Port:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.port_var = tk.StringVar(value="8765")
        ttk.Entry(server_frame, textvariable=self.port_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.server_status_var = tk.StringVar(value="Stopped")
        ttk.Label(server_frame, text="Status:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(server_frame, textvariable=self.server_status_var).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        self.start_server_button = ttk.Button(server_frame, text="Start Server", command=self.start_server)
        self.start_server_button.grid(row=0, column=4, padx=5, pady=5, sticky=tk.E)
        
        self.stop_server_button = ttk.Button(server_frame, text="Stop Server", command=self.stop_server, state=tk.DISABLED)
        self.stop_server_button.grid(row=0, column=5, padx=5, pady=5, sticky=tk.E)
        
        # Agent controls
        agent_frame = ttk.LabelFrame(main_frame, text="Agent Controls", padding="5")
        agent_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Agent controls - row 1
        ttk.Label(agent_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.learning_rate_var = tk.StringVar(value="0.001")
        ttk.Entry(agent_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(agent_frame, text="Epsilon:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.epsilon_var = tk.StringVar(value="1.0")
        ttk.Entry(agent_frame, textvariable=self.epsilon_var, width=10).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Agent controls - row 2
        ttk.Label(agent_frame, text="Episodes:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.episodes_var = tk.StringVar(value="0")
        ttk.Label(agent_frame, textvariable=self.episodes_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(agent_frame, text="Total Steps:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.steps_var = tk.StringVar(value="0")
        ttk.Label(agent_frame, textvariable=self.steps_var).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Agent controls - row 3
        self.agent_status_var = tk.StringVar(value="Not initialized")
        ttk.Label(agent_frame, text="Agent Status:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(agent_frame, textvariable=self.agent_status_var).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.reset_agent_button = ttk.Button(agent_frame, text="Reset Agent", command=self.reset_agent, state=tk.DISABLED)
        self.reset_agent_button.grid(row=2, column=2, padx=5, pady=5, sticky=tk.E)
        
        self.save_agent_button = ttk.Button(agent_frame, text="Save Agent", command=self.save_agent, state=tk.DISABLED)
        self.save_agent_button.grid(row=2, column=3, padx=5, pady=5, sticky=tk.E)
        
        # Agent toggle
        self.agent_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(agent_frame, text="Enable Agent", variable=self.agent_enabled_var, command=self.toggle_agent).grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Game statistics
        stats_frame = ttk.LabelFrame(main_frame, text="Game Statistics", padding="5")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Game statistics - row 1
        ttk.Label(stats_frame, text="Player Health:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.player_health_var = tk.StringVar(value="0/0")
        ttk.Label(stats_frame, textvariable=self.player_health_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Wave:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.wave_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.wave_var).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Game statistics - row 2
        ttk.Label(stats_frame, text="Enemies Nearby:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.enemies_nearby_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.enemies_nearby_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Last Action:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.last_action_var = tk.StringVar(value="None")
        ttk.Label(stats_frame, textvariable=self.last_action_var).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Game statistics - row 3
        ttk.Label(stats_frame, text="Reward:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.reward_var = tk.StringVar(value="0.0")
        ttk.Label(stats_frame, textvariable=self.reward_var).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(stats_frame, text="Total Reward:").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        self.total_reward_var = tk.StringVar(value="0.0")
        ttk.Label(stats_frame, textvariable=self.total_reward_var).grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Console output
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="5")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, width=70, height=15, state=tk.DISABLED)
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Redirect stdout to the console
        self.stdout_redirect = RedirectText(self.console_text)
        sys.stdout = self.stdout_redirect
    
    def start_server(self):
        """Start the WebSocket server."""
        try:
            port = int(self.port_var.get())
            
            # Update UI to show we're working
            print(f"Starting server on port {port}...")
            self.server_status_var.set("Starting...")
            self.start_server_button.config(state=tk.DISABLED)
            self.root.update()  # Force UI update
            
            # Create a console callback function
            def console_callback(message):
                print(message)
                # Update UI directly from the callback
                self.console_text.configure(state="normal")
                self.console_text.insert(tk.END, message)
                self.console_text.see(tk.END)
                self.console_text.configure(state="disabled")
                self.root.update_idletasks()  # Update UI without blocking
            
            # Start server in a separate thread to avoid UI freezing
            import threading
            
            def server_init_thread():
                try:
                    # Initialize the server
                    self.server = WebSocketServer(port, console_callback)
                    
                    # Initialize the agent
                    self.agent = self.server.game_agent
                    
                    # Update agent parameters
                    try:
                        learning_rate = float(self.learning_rate_var.get())
                        epsilon = float(self.epsilon_var.get())
                        self.agent.learning_rate = learning_rate
                        self.agent.epsilon = epsilon
                        print(f"Agent parameters updated: learning_rate={learning_rate}, epsilon={epsilon}")
                    except ValueError:
                        print("Invalid learning rate or epsilon values. Using defaults.")
                    
                    # Start the server
                    success = self.server.start()
                    
                    # Schedule UI updates on the main thread
                    if success:
                        self.root.after(0, self._server_started_success, port)
                    else:
                        self.root.after(0, self._server_started_failure, port)
                        
                except Exception as e:
                    import traceback
                    print(f"Error in server thread: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    self.root.after(0, self._server_error, str(e))
            
            # Start the initialization thread
            init_thread = threading.Thread(target=server_init_thread, daemon=True)
            init_thread.start()
            
        except ValueError:
            messagebox.showerror("Error", "Port must be a number")
            print("Port must be a number")
            self.start_server_button.config(state=tk.NORMAL)
            self.server_status_var.set("Stopped")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error starting server: {e}")
            self.start_server_button.config(state=tk.NORMAL)
            self.server_status_var.set("Stopped")
    
    def _server_started_success(self, port):
        """Called when server successfully starts."""
        self.server_status_var.set("Running")
        self.stop_server_button.config(state=tk.NORMAL)
        self.reset_agent_button.config(state=tk.NORMAL)
        self.save_agent_button.config(state=tk.NORMAL)
        self.agent_status_var.set("Ready")
        print(f"Server started on port {port}")
    
    def _server_started_failure(self, port):
        """Called when server fails to start."""
        self.start_server_button.config(state=tk.NORMAL)
        self.server_status_var.set("Stopped")
        messagebox.showerror("Error", f"Failed to start server on port {port}")
        print(f"Failed to start server on port {port}")
    
    def _server_error(self, error_message):
        """Called when there's an exception during server startup."""
        self.start_server_button.config(state=tk.NORMAL)
        self.server_status_var.set("Error")
        messagebox.showerror("Server Error", error_message)
        print(f"Server error: {error_message}")
    
    def stop_server(self):
        """Stop the WebSocket server."""
        if not self.server:
            return
        
        # Update UI to show we're working
        self.server_status_var.set("Stopping...")
        self.stop_server_button.config(state=tk.DISABLED)
        self.root.update()  # Force UI update
        
        # Stop server in a separate thread to avoid UI freezing
        import threading
        
        def server_stop_thread():
            try:
                success = self.server.stop()
                # Schedule UI updates on the main thread
                if success:
                    self.root.after(0, self._server_stopped_success)
                else:
                    self.root.after(0, self._server_stopped_failure)
            except Exception as e:
                import traceback
                print(f"Error stopping server: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                self.root.after(0, self._server_stopped_error, str(e))
        
        # Start the stop thread
        stop_thread = threading.Thread(target=server_stop_thread, daemon=True)
        stop_thread.start()
    
    def _server_stopped_success(self):
        """Called when server successfully stops."""
        self.server_status_var.set("Stopped")
        self.start_server_button.config(state=tk.NORMAL)
        self.stop_server_button.config(state=tk.DISABLED)
        self.agent_status_var.set("Not initialized")
        print("Server stopped")
    
    def _server_stopped_failure(self):
        """Called when server fails to stop."""
        self.server_status_var.set("Running")
        self.stop_server_button.config(state=tk.NORMAL)
        messagebox.showerror("Error", "Failed to stop server")
        print("Failed to stop server")
    
    def _server_stopped_error(self, error_message):
        """Called when there's an exception during server stop."""
        self.server_status_var.set("Error")
        self.start_server_button.config(state=tk.NORMAL)
        self.stop_server_button.config(state=tk.NORMAL)
        messagebox.showerror("Server Stop Error", error_message)
        print(f"Server stop error: {error_message}")
    
    def reset_agent(self):
        """Reset the agent."""
        if self.agent:
            # Update agent parameters
            try:
                learning_rate = float(self.learning_rate_var.get())
                epsilon = float(self.epsilon_var.get())
                self.agent.learning_rate = learning_rate
                self.agent.epsilon = epsilon
            except ValueError:
                pass
            
            # Reset the agent
            self.agent.reset()
            
            # Update UI
            self.episodes_var.set("0")
            self.steps_var.set("0")
            self.agent_status_var.set("Reset")
            print("Agent reset")
    
    def save_agent(self):
        """Save the agent model."""
        if self.agent:
            self.agent.save_model()
            print("Agent model saved")
    
    def toggle_agent(self):
        """Toggle the agent on/off."""
        if self.server:
            enabled = self.agent_enabled_var.get()
            self.server.toggle_agent(enabled)
            self.agent_status_var.set("Active" if enabled else "Disabled")
            print(f"Agent {'enabled' if enabled else 'disabled'}")
    
    def update_ui(self):
        """Update the UI with current statistics."""
        try:
            if not hasattr(self, '_last_update_time'):
                self._last_update_time = time.time()
                self._update_full = True
            else:
                # Only do a full update every 2 seconds to save CPU
                current_time = time.time()
                self._update_full = (current_time - self._last_update_time) >= 2.0
                if self._update_full:
                    self._last_update_time = current_time
            
            if self.agent:
                # Basic updates every time
                self.episodes_var.set(str(self.agent.episode_count))
                self.steps_var.set(str(self.agent.step_count))
                
                # Only update epsilon on full updates (it changes slowly)
                if self._update_full:
                    self.epsilon_var.set(f"{self.agent.epsilon:.4f}")
                
                # Update game statistics if we have game state
                if hasattr(self.agent, "last_game_state") and self.agent.last_game_state:
                    game_state = self.agent.last_game_state
                    
                    # Health updates every cycle (important)
                    if "player" in game_state and "health" in game_state["player"]:
                        health = game_state["player"]["health"]
                        current_health = f"{health['current']}/{health['max']}"
                        if self.player_health_var.get() != current_health:
                            self.player_health_var.set(current_health)
                    
                    # These are less critical, update on full refresh
                    if self._update_full:
                        if "game" in game_state and "wave" in game_state["game"]:
                            self.wave_var.set(str(game_state["game"]["wave"]))
                        
                        if "nearby" in game_state and "enemies" in game_state["nearby"]:
                            self.enemies_nearby_var.set(str(len(game_state["nearby"]["enemies"])))
                
                # Update reward information (important feedback)
                self.reward_var.set(f"{self.agent.cumulative_reward:.2f}")
                
                # Last action can change frequently, update always
                if self.agent.last_action is not None:
                    action_names = {
                        0: "None",
                        1: "Up",
                        2: "Down",
                        3: "Left",
                        4: "Right",
                        5: "Up-Left",
                        6: "Up-Right",
                        7: "Down-Left",
                        8: "Down-Right"
                    }
                    action_name = action_names.get(self.agent.last_action, "Unknown")
                    if self.last_action_var.get() != action_name:
                        self.last_action_var.set(action_name)
            
        except Exception as e:
            import traceback
            print(f"Error updating UI: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            # Schedule the next update (slightly longer interval for better performance)
            self.update_timer = self.root.after(1000, self.update_ui)
    
    def on_close(self):
        """Handle window close event."""
        if self.server and self.server.running:
            if messagebox.askyesno("Confirm Exit", "The server is still running. Do you want to stop it and exit?"):
                self.stop_server()
                self.cleanup()
                self.root.destroy()
        else:
            self.cleanup()
            self.root.destroy()
    
    def cleanup(self):
        """Clean up resources before exit."""
        # Cancel the update timer
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Save agent model if it exists
        if self.agent:
            try:
                self.agent.save_model()
                print("Agent model saved")
            except Exception as e:
                print(f"Error saving agent model: {e}")


def main():
    """Main entry point for the application."""
    # Make sure models directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    
    # Add debug logging
    print("Python version:", sys.version)
    print("PyTorch device available:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Current directory:", os.getcwd())
    print("Available models:", os.listdir(os.path.join(os.path.dirname(__file__), "models")))
    
    # Start the UI
    print("Starting Deepotato AI Bot Controller")
    root = tk.Tk()
    app = DeepLearningBotUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
