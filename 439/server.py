import asyncio
import json
import logging
import time
import sqlite3
from contextlib import contextmanager
import websockets
from websockets.server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 30
HEARTBEAT_TIMEOUT = 10
MAX_OFFLINE_MESSAGES = 100
DB_PATH = "chat_server.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS offline_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                to_user TEXT NOT NULL,
                from_user TEXT,
                room TEXT,
                message_type TEXT,
                content TEXT,
                timestamp REAL,
                delivered INTEGER DEFAULT 0
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_offline_user ON offline_messages(to_user, delivered)
        ''')
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def save_offline_message(to_user, from_user, room, message_type, content):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO offline_messages (to_user, from_user, room, message_type, content, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (to_user, from_user, room, message_type, content, time.time())
        )
        conn.commit()


def get_offline_messages(username):
    with get_db() as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT id, from_user, room, message_type, content, timestamp FROM offline_messages WHERE to_user = ? AND delivered = 0 ORDER BY timestamp ASC",
            (username,)
        )
        rows = c.fetchall()
        return [dict(row) for row in rows]


def mark_messages_delivered(msg_ids):
    if not msg_ids:
        return
    with get_db() as conn:
        c = conn.cursor()
        placeholders = ",".join("?" * len(msg_ids))
        c.execute(
            f"UPDATE offline_messages SET delivered = 1 WHERE id IN ({placeholders})",
            msg_ids
        )
        conn.commit()


def purge_old_offline_messages(username):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id FROM offline_messages WHERE to_user = ? AND delivered = 0 ORDER BY timestamp DESC LIMIT -1 OFFSET ?",
            (username, MAX_OFFLINE_MESSAGES)
        )
        rows = c.fetchall()
        if rows:
            ids = [row[0] for row in rows]
            placeholders = ",".join("?" * len(ids))
            c.execute(f"DELETE FROM offline_messages WHERE id IN ({placeholders})", ids)
            conn.commit()


class Room:
    def __init__(self, name):
        self.name = name
        self.members: set[websockets.WebSocketServerProtocol] = set()
        self.history: list[dict] = []

    def add(self, ws):
        self.members.add(ws)

    def remove(self, ws):
        self.members.discard(ws)

    async def broadcast(self, message_obj, exclude=None, persist=False):
        dead = []
        msg_json = json.dumps(message_obj)
        if persist:
            self.history.append({**message_obj, "timestamp": time.time()})
            if len(self.history) > 100:
                self.history = self.history[-100:]
        for member in list(self.members):
            if member == exclude:
                continue
            try:
                await member.send(msg_json)
            except Exception as e:
                logger.warning(f"Failed to send to client in room {self.name}: {e}")
                dead.append(member)
        for member in dead:
            self.members.discard(member)


class ChatServer:
    def __init__(self):
        self.rooms: dict[str, Room] = {}
        self.client_rooms: dict[websockets.WebSocketServerProtocol, set[str]] = {}
        self.client_names: dict[websockets.WebSocketServerProtocol, str] = {}
        self.name_to_client: dict[str, websockets.WebSocketServerProtocol] = {}
        init_db()

    def _get_or_create_room(self, name) -> Room:
        if name not in self.rooms:
            self.rooms[name] = Room(name)
        return self.rooms[name]

    def _get_online_users(self):
        return [
            {
                "name": name,
                "rooms": list(self.client_rooms.get(ws, set()))
            }
            for name, ws in self.name_to_client.items()
        ]

    def _get_stats(self):
        return {
            "online_count": len(self.client_names),
            "room_count": len(self.rooms),
            "rooms": [
                {"name": r.name, "members": len(r.members)}
                for r in self.rooms.values()
            ],
            "online_users": self._get_online_users()
        }

    async def _join_room(self, ws, room_name):
        room = self._get_or_create_room(room_name)
        room.add(ws)
        if ws not in self.client_rooms:
            self.client_rooms[ws] = set()
        self.client_rooms[ws].add(room_name)
        username = self.client_names.get(ws, "anonymous")
        await room.broadcast(
            {
                "type": "system",
                "message": f"[{username}] joined room [{room_name}]",
                "room": room_name,
                "timestamp": time.time(),
            },
            exclude=ws,
            persist=True,
        )
        if ws.open:
            try:
                await ws.send(json.dumps({
                    "type": "system",
                    "message": f"You joined room [{room_name}]. Members: {len(room.members)}",
                    "room": room_name,
                    "history": room.history[-50:],
                }))
            except Exception as e:
                logger.warning(f"Failed to send join confirmation to {id(ws)}: {e}")

    async def _leave_room(self, ws, room_name, notify_self=True):
        if room_name not in self.rooms:
            if notify_self and ws.open:
                try:
                    await ws.send(json.dumps({
                        "type": "error",
                        "message": f"Room [{room_name}] does not exist",
                    }))
                except Exception:
                    pass
            return
        room = self.rooms[room_name]
        room.remove(ws)
        if ws in self.client_rooms:
            self.client_rooms[ws].discard(room_name)
        username = self.client_names.get(ws, "anonymous")
        await room.broadcast(
            {
                "type": "system",
                "message": f"[{username}] left room [{room_name}]",
                "room": room_name,
                "timestamp": time.time(),
            },
            persist=True,
        )
        if notify_self and ws.open:
            try:
                await ws.send(json.dumps({
                    "type": "system",
                    "message": f"You left room [{room_name}]",
                    "room": room_name,
                }))
            except Exception:
                pass
        if not room.members:
            del self.rooms[room_name]

    async def _safe_send(self, ws, message):
        if not ws.open:
            return False
        try:
            await ws.send(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {id(ws)}: {e}")
            return False

    async def _deliver_offline_messages(self, ws, username):
        messages = get_offline_messages(username)
        if not messages:
            return
        msg_ids = []
        for msg in messages:
            msg_ids.append(msg["id"])
            await self._safe_send(ws, json.dumps({
                "type": "offline",
                "from": msg["from_user"],
                "room": msg["room"],
                "message_type": msg["message_type"],
                "message": msg["content"],
                "timestamp": msg["timestamp"],
            }))
        mark_messages_delivered(msg_ids)
        logger.info(f"Delivered {len(msg_ids)} offline messages to {username}")

    async def _handle_unicast(self, ws, data):
        from_user = self.client_names.get(ws, "anonymous")
        to_user = data.get("to")
        text = data.get("message", "")
        if not to_user:
            await self._safe_send(ws, json.dumps({
                "type": "error", "message": "Missing 'to' field for unicast"
            }))
            return
        target_ws = self.name_to_client.get(to_user)
        msg_obj = {
            "type": "unicast",
            "from": from_user,
            "to": to_user,
            "message": text,
            "timestamp": time.time(),
        }
        if target_ws and target_ws.open:
            success = await self._safe_send(target_ws, json.dumps(msg_obj))
            if not success:
                save_offline_message(to_user, from_user, None, "unicast", text)
        else:
            save_offline_message(to_user, from_user, None, "unicast", text)
            purge_old_offline_messages(to_user)
        await self._safe_send(ws, json.dumps({
            "type": "ack",
            "to": to_user,
            "message": text,
            "delivered": target_ws is not None and target_ws.open,
        }))

    async def _handle_multicast(self, ws, data):
        from_user = self.client_names.get(ws, "anonymous")
        room_name = data.get("room")
        text = data.get("message", "")
        if not room_name:
            await self._safe_send(ws, json.dumps({
                "type": "error", "message": "Missing 'room' field for multicast"
            }))
            return
        if room_name not in self.client_rooms.get(ws, set()):
            await self._safe_send(ws, json.dumps({
                "type": "error",
                "message": f"You are not in room [{room_name}]. Join it first.",
            }))
            return
        room = self._get_or_create_room(room_name)
        msg_obj = {
            "type": "multicast",
            "from": from_user,
            "room": room_name,
            "message": text,
            "timestamp": time.time(),
        }
        online_users = set()
        dead = []
        msg_json = json.dumps(msg_obj)
        for member in list(room.members):
            if member == ws:
                continue
            online_users.add(self.client_names.get(member, "anonymous"))
            try:
                await member.send(msg_json)
            except Exception as e:
                logger.warning(f"Failed to multicast to {id(member)}: {e}")
                dead.append(member)
        for member in dead:
            room.members.discard(member)
        all_room_users = set()
        for client_ws, rooms in self.client_rooms.items():
            if room_name in rooms:
                all_room_users.add(self.client_names.get(client_ws, "anonymous"))
        offline_users = all_room_users - online_users - {from_user}
        for offline_user in offline_users:
            save_offline_message(offline_user, from_user, room_name, "multicast", text)
            purge_old_offline_messages(offline_user)
        await self._safe_send(ws, json.dumps({
            "type": "ack",
            "room": room_name,
            "message": text,
            "online_receivers": len(online_users),
            "offline_receivers": len(offline_users),
        }))

    async def _handle_message(self, ws, data: dict):
        msg_type = data.get("type")

        if msg_type == "setName":
            name = data.get("name", "anonymous")
            old_name = self.client_names.get(ws)
            if old_name:
                self.name_to_client.pop(old_name, None)
            self.client_names[ws] = name
            self.name_to_client[name] = ws
            await self._safe_send(ws, json.dumps({
                "type": "system",
                "message": f"Your name is set to [{name}]",
            }))
            asyncio.create_task(self._deliver_offline_messages(ws, name))

        elif msg_type == "join":
            room_name = data.get("room", "lobby")
            await self._join_room(ws, room_name)

        elif msg_type == "leave":
            room_name = data.get("room")
            if not room_name:
                await self._safe_send(ws, json.dumps({
                    "type": "error", "message": "Missing 'room' field"
                }))
                return
            await self._leave_room(ws, room_name)

        elif msg_type == "message":
            room_name = data.get("room", "lobby")
            text = data.get("message", "")
            username = self.client_names.get(ws, "anonymous")
            if room_name not in self.client_rooms.get(ws, set()):
                await self._safe_send(ws, json.dumps({
                    "type": "error",
                    "message": f"You are not in room [{room_name}]. Join it first.",
                }))
                return
            room = self._get_or_create_room(room_name)
            await room.broadcast(
                {
                    "type": "message",
                    "from": username,
                    "message": text,
                    "room": room_name,
                    "timestamp": time.time(),
                },
                exclude=ws,
                persist=True,
            )
            await self._safe_send(ws, json.dumps({
                "type": "ack",
                "message": text,
                "room": room_name,
            }))

        elif msg_type == "unicast":
            await self._handle_unicast(ws, data)

        elif msg_type == "multicast":
            await self._handle_multicast(ws, data)

        elif msg_type == "listRooms":
            room_list = [
                {"name": r.name, "members": len(r.members)}
                for r in self.rooms.values()
            ]
            await self._safe_send(ws, json.dumps({
                "type": "roomList", "rooms": room_list
            }))

        elif msg_type == "listMembers":
            room_name = data.get("room")
            if not room_name or room_name not in self.rooms:
                await self._safe_send(ws, json.dumps({
                    "type": "error", "message": "Room not found"
                }))
                return
            room = self.rooms[room_name]
            members = [
                self.client_names.get(m, "anonymous") for m in room.members
            ]
            await self._safe_send(ws, json.dumps({
                "type": "memberList", "room": room_name, "members": members
            }))

        elif msg_type == "listUsers":
            await self._safe_send(ws, json.dumps({
                "type": "userList",
                "users": self._get_online_users()
            }))

        elif msg_type == "stats":
            await self._safe_send(ws, json.dumps({
                "type": "stats",
                **self._get_stats()
            }))

        else:
            await self._safe_send(ws, json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            }))

    async def _cleanup(self, ws):
        try:
            username = self.client_names.get(ws)
            if username:
                self.name_to_client.pop(username, None)
            if ws in self.client_rooms:
                for room_name in list(self.client_rooms[ws]):
                    await self._leave_room(ws, room_name, notify_self=False)
                if ws in self.client_rooms:
                    del self.client_rooms[ws]
            self.client_names.pop(ws, None)
            logger.info(f"Cleaned up connection {id(ws)} ({username})")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    async def _heartbeat(self, ws):
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if not ws.open:
                    break
                try:
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=HEARTBEAT_TIMEOUT)
                    logger.debug(f"Heartbeat OK for {id(ws)}")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Heartbeat timeout for {id(ws)} after {HEARTBEAT_TIMEOUT}s, closing"
                    )
                    await ws.close(code=1008, reason="Heartbeat timeout")
                    break
                except Exception as e:
                    logger.info(f"Connection {id(ws)} closed during heartbeat: {e}")
                    break
        except asyncio.CancelledError:
            logger.debug(f"Heartbeat cancelled for {id(ws)}")
            raise
        except Exception as e:
            logger.error(f"Heartbeat error for {id(ws)}: {e}", exc_info=True)

    async def handler(self, ws):
        logger.info(f"New connection {id(ws)} from {ws.remote_address}")
        heartbeat_task = None
        try:
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    if ws.open:
                        try:
                            await ws.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                        except Exception:
                            pass
                    continue
                try:
                    await self._handle_message(ws, data)
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)
                    if ws.open:
                        try:
                            await ws.send(json.dumps({"type": "error", "message": "Internal server error"}))
                        except Exception:
                            pass
        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError):
            logger.info(f"Connection {id(ws)} closed normally")
        except asyncio.CancelledError:
            logger.info(f"Connection {id(ws)} cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in handler for {id(ws)}: {e}", exc_info=True)
        finally:
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"Error cancelling heartbeat: {e}")
            await self._cleanup(ws)


async def main(host="0.0.0.0", port=8765):
    server = ChatServer()
    logger.info(f"WebSocket Chat Server starting on ws://{host}:{port}")
    async with serve(
        server.handler,
        host,
        port,
        ping_interval=HEARTBEAT_INTERVAL,
        ping_timeout=HEARTBEAT_TIMEOUT,
        close_timeout=5,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
