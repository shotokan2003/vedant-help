"""
WebSocket ping implementation to keep connections alive
"""
from fastapi import WebSocket
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
        self.ping_interval = 30  # seconds

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        asyncio.create_task(self._keep_alive(client_id))
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def _keep_alive(self, client_id: str):
        """Send periodic pings to keep WebSocket connection alive"""
        try:
            while client_id in self.active_connections:
                await asyncio.sleep(self.ping_interval)
                websocket = self.active_connections.get(client_id)
                if websocket:
                    try:
                        # Send ping
                        await websocket.send_text("ping")
                        print(f"Sent ping to client: {client_id}")
                    except Exception as e:
                        print(f"Error sending ping to {client_id}: {e}")
                        self.disconnect(client_id)
                        break
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            self.disconnect(client_id)
