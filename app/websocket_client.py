"""
WebSocket client implementation for testing the WebSocket server.
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Optional

import websockets


class WebSocketClient:
    """A WebSocket client for testing connections to the server."""

    def __init__(
        self, host: str, port: int, console_callback: Optional[Callable] = None
    ):
        """Initialize the WebSocket client.

        Args:
            host: The host to connect to
            port: The port to connect to
            console_callback: A function to call with log messages
        """
        self.host = host
        self.port = port
        self.console_callback = console_callback
        self.client_thread = None
        self.event_loop = None

        # Set up logging
        self.logger = logging.getLogger(__name__)
        if console_callback:
            self.logger.addHandler(ConsoleCallbackHandler(console_callback))
            self.logger.setLevel(logging.INFO)

    def send_test_message(self, message: str) -> bool:
        """Send a test message to the WebSocket server.

        Args:
            message: The message to send

        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        # Start a new connection, send the message, and close
        try:
            # Create a new thread for this client session
            self.client_thread = threading.Thread(
                target=self._run_client_thread, args=(message,)
            )
            self.client_thread.daemon = True
            self.client_thread.start()
            return True
        except Exception as e:
            self._log(f"Error sending test message: {str(e)}")
            return False

    def _run_client_thread(self, message: str):
        """Run a client session in its own thread with its own event loop."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.event_loop = loop

            # Run the client session
            loop.run_until_complete(self._client_session(message))
        except Exception as e:
            self._log(f"Client thread error: {str(e)}")
        finally:
            # Clean up
            if loop and not loop.is_closed():
                loop.close()
            self._log("Client thread terminated")

    async def _client_session(self, message: str):
        """Connect to the server, send a message, and receive the response."""
        uri = f"ws://{self.host}:{self.port}"
        self._log(f"Connecting to {uri}...")

        try:
            async with websockets.connect(uri) as websocket:
                self._log(f"Connected to server at {uri}")

                # Send the message
                self._log(f"Sending message: {message}")
                await websocket.send(message)

                # Wait for response
                self._log("Waiting for response...")
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    self._log(f"Received response: {response}")
                except asyncio.TimeoutError:
                    self._log("Timeout waiting for response")

                # Properly close the connection
                await websocket.close()
                self._log("Connection closed")

        except ConnectionRefusedError:
            self._log(f"Connection refused to {uri}. Is the server running?")
        except websockets.exceptions.InvalidStatusCode as e:
            self._log(f"Invalid status code: {e}")
        except Exception as e:
            self._log(f"Error during client session: {str(e)}")

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
            self.callback(f"Client: {msg}\n")
        except Exception:
            self.handleError(record)
