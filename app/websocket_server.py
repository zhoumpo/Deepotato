"""
WebSocket server implementation using the websockets library.
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Optional, Set

import websockets


class WebSocketServer:
    """A WebSocket server that can be started and stopped from a GUI."""

    def __init__(self, port: int, console_callback: Optional[Callable] = None):
        """Initialize the WebSocket server.

        Args:
            port: The port to run the server on
            console_callback: A function to call with log messages
        """
        self.port = port
        self.console_callback = console_callback
        self.server = None
        self.server_thread = None
        self.event_loop = None
        self.running = False
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self._shutdown_event = threading.Event()

        # Set up logging to use our console callback
        self.logger = logging.getLogger(__name__)
        if console_callback:
            self.logger.addHandler(ConsoleCallbackHandler(console_callback))
            self.logger.setLevel(logging.INFO)

    def start(self) -> bool:
        """Start the WebSocket server in a separate thread.

        Returns:
            bool: True if the server started successfully, False otherwise
        """
        if self.running:
            self._log("Server is already running")
            return True

        # Reset shutdown event if previously set
        self._shutdown_event.clear()

        try:
            self._log("Starting server thread...")
            # Create a thread that will run our event loop
            self.server_thread = threading.Thread(target=self._run_server_thread)
            self.server_thread.daemon = True  # Thread will exit when main program exits
            self.server_thread.start()

            # Wait briefly to ensure the server starts successfully
            start_time = time.time()
            while not self.running and time.time() - start_time < 2.0:
                time.sleep(0.1)  # Check every 100ms for up to 2 seconds

            return self.running
        except Exception as e:
            self._log(f"Error starting server thread: {str(e)}")
            return False

    def stop(self) -> bool:
        """Stop the WebSocket server.

        Returns:
            bool: True if the server stopped successfully, False otherwise
        """
        if not self.running:
            return False

        try:
            # Signal the server thread to shut down
            self._shutdown_event.set()

            # Wait for the server thread to finish
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5.0)

            self.running = False
            self._log("Server stopped")
            return True
        except Exception as e:
            self._log(f"Error stopping server: {str(e)}")
            return False

    def _run_server_thread(self):
        """Entry point for the server thread. Sets up and runs the asyncio event loop."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.event_loop = loop

            # Run the server in the event loop
            loop.run_until_complete(self._run_server())
        except Exception as e:
            self._log(f"Server thread error: {str(e)}")
        finally:
            self._log("Server thread exiting")
            self.running = False

    async def _run_server(self):
        """Async coroutine that runs the actual WebSocket server."""
        try:
            # Create and start the server
            self._log(f"Starting WebSocket server on port {self.port}")

            # It's important to bind to 0.0.0.0 to allow external connections
            # Change to 'localhost' if you only want local connections
            self.server = await websockets.serve(
                self._handle_client_connection,
                "0.0.0.0",  # Changed from localhost to allow connections from any IP
                self.port,
            )

            self.running = True
            self._log(f"WebSocket server running on port {self.port}")

            # Keep the server running until we get a shutdown signal
            while not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)

            # Clean shutdown
            self._log("Server shutdown initiated")

            # Close client connections
            if self.connected_clients:
                self._log(f"Closing {len(self.connected_clients)} client connections")
                close_tasks = [client.close() for client in self.connected_clients]
                await asyncio.gather(*close_tasks, return_exceptions=True)
                self.connected_clients.clear()

            # Close the server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self._log("Server closed")

        except Exception as e:
            self._log(f"Error in server: {str(e)}")
        finally:
            self.running = False

            # Close client connections
            if self.connected_clients:
                self._log(f"Closing {len(self.connected_clients)} client connections")
                close_tasks = [client.close() for client in self.connected_clients]
                await asyncio.gather(*close_tasks, return_exceptions=True)
                self.connected_clients.clear()

            # Close the server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self._log("Server closed")

    async def _handle_client_connection(self, websocket):
        """Handle a client connection to the WebSocket server.

        Note: Recent versions of websockets library don't pass 'path' parameter anymore.
        """
        client_id = id(websocket)
        remote_address = (
            websocket.remote_address
            if hasattr(websocket, "remote_address")
            else "unknown"
        )
        self._log(f"Client connected: {client_id} from {remote_address}")

        # Add to connected clients set
        self.connected_clients.add(websocket)

        try:
            # Handle messages from the client
            async for message in websocket:
                self._log(f"Received message from client {client_id}: {message}")

                # Echo the message back to the client
                await websocket.send(f"Echo: {message}")

        except websockets.exceptions.ConnectionClosed as e:
            self._log(f"Client {client_id} disconnected: {e}")
        except Exception as e:
            self._log(f"Error handling client {client_id}: {e}")
        finally:
            # Remove from connected clients set
            self.connected_clients.discard(websocket)

    def _log(self, message: str):
        """Log a message to the console callback if available."""
        self.logger.info(message)


class ConsoleCallbackHandler(logging.Handler):
    """A logging handler that forwards log messages to a console callback."""

    def __init__(self, callback: Callable):
        """Initialize with a callback function."""
        super().__init__()
        self.callback = callback

    def emit(self, record):
        """Emit a log record by calling the callback function."""
        try:
            msg = self.format(record)
            self.callback(f"{msg}\n")
        except Exception:
            self.handleError(record)
