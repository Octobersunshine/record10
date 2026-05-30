const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const path = require("path");

app.use(express.static(path.join(__dirname, "public")));

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] },
});

const rooms = new Map();

function getRoom(roomId) {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, { users: new Map() });
  }
  return rooms.get(roomId);
}

function leaveRoom(socket) {
  const roomId = socket.data.roomId;
  if (!roomId) return;

  const room = rooms.get(roomId);
  if (!room) return;

  const userId = socket.data.userId;
  room.users.delete(userId);

  socket.to(roomId).emit("user-left", { userId });

  if (room.users.size === 0) {
    rooms.delete(roomId);
  }

  socket.leave(roomId);
  socket.data.roomId = null;
  socket.data.userId = null;
}

io.on("connection", (socket) => {
  console.log(`[CONNECT] ${socket.id}`);

  socket.on("join-room", ({ roomId, userId }, callback) => {
    if (!roomId || !userId) {
      return callback?.({ success: false, error: "roomId and userId are required" });
    }

    const room = getRoom(roomId);

    if (room.users.size >= 2) {
      return callback?.({ success: false, error: "Room is full (max 2 users)" });
    }

    if (room.users.has(userId)) {
      return callback?.({ success: false, error: "userId already taken in this room" });
    }

    socket.data.roomId = roomId;
    socket.data.userId = userId;

    room.users.set(userId, socket.id);
    socket.join(roomId);

    const existingUsers = [];
    room.users.forEach((sid, uid) => {
      if (uid !== userId) {
        existingUsers.push({ userId: uid, socketId: sid });
      }
    });

    socket.to(roomId).emit("user-joined", { userId, socketId: socket.id });

    callback?.({
      success: true,
      users: existingUsers,
    });

    console.log(`[JOIN] ${userId} joined room ${roomId} (${room.users.size} users)`);
  });

  socket.on("offer", ({ targetUserId, sdp }) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;

    const room = rooms.get(roomId);
    if (!room) return;

    const targetSocketId = room.users.get(targetUserId);
    if (!targetSocketId) return;

    io.to(targetSocketId).emit("offer", {
      fromUserId: socket.data.userId,
      sdp,
    });

    console.log(`[OFFER] ${socket.data.userId} -> ${targetUserId} in room ${roomId}`);
  });

  socket.on("answer", ({ targetUserId, sdp }) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;

    const room = rooms.get(roomId);
    if (!room) return;

    const targetSocketId = room.users.get(targetUserId);
    if (!targetSocketId) return;

    io.to(targetSocketId).emit("answer", {
      fromUserId: socket.data.userId,
      sdp,
    });

    console.log(`[ANSWER] ${socket.data.userId} -> ${targetUserId} in room ${roomId}`);
  });

  socket.on("ice-candidate", ({ targetUserId, candidate }) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;

    const room = rooms.get(roomId);
    if (!room) return;

    const targetSocketId = room.users.get(targetUserId);
    if (!targetSocketId) return;

    io.to(targetSocketId).emit("ice-candidate", {
      fromUserId: socket.data.userId,
      candidate,
    });
  });

  socket.on("leave-room", () => {
    const userId = socket.data.userId;
    const roomId = socket.data.roomId;
    leaveRoom(socket);
    if (userId && roomId) {
      console.log(`[LEAVE] ${userId} left room ${roomId}`);
    }
  });

  socket.on("disconnect", (reason) => {
    const userId = socket.data.userId;
    const roomId = socket.data.roomId;
    leaveRoom(socket);
    if (userId && roomId) {
      console.log(`[DISCONNECT] ${userId} from room ${roomId} (reason: ${reason})`);
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`WebRTC Signaling Server running on http://localhost:${PORT}`);
});
