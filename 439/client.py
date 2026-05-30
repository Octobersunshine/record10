import asyncio
import json
import sys
import websockets


async def client(uri: str, name: str):
    try:
        async with websockets.connect(uri, close_timeout=5) as ws:
            await ws.send(json.dumps({"type": "setName", "name": name}))

            async def receiver():
                try:
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            print(f"\r!!! Invalid JSON received !!!")
                            print("> ", end="", flush=True)
                            continue
                        msg_type = data.get("type")
                        if msg_type == "message":
                            print(f"\r[{data['room']}] {data['from']}: {data['message']}")
                        elif msg_type == "system":
                            print(f"\r*** {data['message']} ***")
                        elif msg_type == "unicast":
                            print(f"\r[PM] {data['from']}: {data['message']}")
                        elif msg_type == "multicast":
                            print(f"\r[MCAST {data['room']}] {data['from']}: {data['message']}")
                        elif msg_type == "offline":
                            prefix = "[OFFLINE PM]" if data.get("message_type") == "unicast" else f"[OFFLINE {data.get('room', '?')}]"
                            print(f"\r{prefix} {data['from']}: {data['message']}")
                        elif msg_type == "ack":
                            pass
                        elif msg_type == "error":
                            print(f"\r!!! {data['message']} !!!")
                        elif msg_type == "roomList":
                            print("\r--- Rooms ---")
                            for r in data["rooms"]:
                                print(f"  {r['name']} ({r['members']} members)")
                        elif msg_type == "memberList":
                            print(f"\r--- Members in {data['room']} ---")
                            for m in data["members"]:
                                print(f"  {m}")
                        elif msg_type == "userList":
                            print("\r--- Online Users ---")
                            for u in data["users"]:
                                rooms = ", ".join(u.get("rooms", [])) or "(none)"
                                print(f"  {u['name']} - Rooms: {rooms}")
                        elif msg_type == "stats":
                            print("\r--- Server Stats ---")
                            print(f"  Online users: {data['online_count']}")
                            print(f"  Active rooms: {data['room_count']}")
                            print(f"  Rooms:")
                            for r in data.get("rooms", []):
                                print(f"    {r['name']} ({r['members']} members)")
                        print("> ", end="", flush=True)
                except websockets.exceptions.ConnectionClosed:
                    print("\r*** Connection closed by server ***")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"\r!!! Receiver error: {e} !!!")

            recv_task = asyncio.create_task(receiver())

            print("Commands:")
            print("  /join <room>              - Join a room")
            print("  /leave <room>             - Leave a room")
            print("  /rooms                    - List all rooms")
            print("  /members <room>           - List members in a room")
            print("  /users                    - List all online users")
            print("  /stats                    - Show server stats")
            print("  /whisper <user> <message> - Send private message")
            print("  /mcast <room>:<message>   - Send multicast to room (with offline storage)")
            print("  /quit                     - Quit")
            print()
            print("  Default: room:message      - Broadcast message to room")
            print()
            print("> ", end="", flush=True)

            loop = asyncio.get_event_loop()
            try:
                while True:
                    if recv_task.done():
                        break
                    line = await loop.run_in_executor(
                        None, lambda: sys.stdin.readline().strip()
                    )
                    if not line:
                        continue

                    if line.startswith("/join "):
                        room = line[6:].strip()
                        await ws.send(json.dumps({"type": "join", "room": room}))
                    elif line.startswith("/leave "):
                        room = line[7:].strip()
                        await ws.send(json.dumps({"type": "leave", "room": room}))
                    elif line == "/rooms":
                        await ws.send(json.dumps({"type": "listRooms"}))
                    elif line.startswith("/members "):
                        room = line[9:].strip()
                        await ws.send(json.dumps({"type": "listMembers", "room": room}))
                    elif line == "/users":
                        await ws.send(json.dumps({"type": "listUsers"}))
                    elif line == "/stats":
                        await ws.send(json.dumps({"type": "stats"}))
                    elif line.startswith("/whisper "):
                        parts = line[9:].strip().split(" ", 1)
                        if len(parts) < 2:
                            print("\r!!! Usage: /whisper <user> <message> !!!")
                            print("> ", end="", flush=True)
                            continue
                        target_user, message = parts[0], parts[1]
                        await ws.send(json.dumps({
                            "type": "unicast", "to": target_user, "message": message
                        }))
                    elif line.startswith("/mcast "):
                        rest = line[7:].strip()
                        parts = rest.split(":", 1)
                        if len(parts) < 2:
                            print("\r!!! Usage: /mcast <room>:<message> !!!")
                            print("> ", end="", flush=True)
                            continue
                        room, message = parts[0].strip(), parts[1].strip()
                        await ws.send(json.dumps({
                            "type": "multicast", "room": room, "message": message
                        }))
                    elif line == "/quit":
                        break
                    else:
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            room, message = parts[0].strip(), parts[1].strip()
                        else:
                            room, message = "lobby", line.strip()
                        await ws.send(json.dumps({
                            "type": "message", "room": room, "message": message
                        }))

                    print("> ", end="", flush=True)
            except websockets.exceptions.ConnectionClosed:
                print("\r*** Connection closed ***")
            finally:
                if not recv_task.done():
                    recv_task.cancel()
                    try:
                        await recv_task
                    except (asyncio.CancelledError, Exception):
                        pass
    except websockets.exceptions.ConnectionClosed:
        print("*** Connection closed by server ***")
    except websockets.exceptions.InvalidURI:
        print(f"!!! Invalid URI: {uri} !!!")
    except OSError as e:
        print(f"!!! Connection failed: {e} !!!")
    except Exception as e:
        print(f"!!! Error: {e} !!!")


if __name__ == "__main__":
    server_uri = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
    username = sys.argv[2] if len(sys.argv) > 2 else "user"
    asyncio.run(client(server_uri, username))
